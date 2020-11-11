#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <mutex>
#include <algorithm>
#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

// #define VERBOSE 1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    #pragma omp parallel
    {
        std::vector<int> partial_frontier;
        #pragma omp for                                                        
        for (int i=0; i<frontier->count; i++) {

            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                int curr_dst = distances[outgoing];
                if (curr_dst != NOT_VISITED_MARKER) continue;
                if (!__sync_bool_compare_and_swap(&distances[outgoing], curr_dst, distances[node] + 1)) continue;
                partial_frontier.push_back(outgoing);                
            }
        }
        int index = __sync_fetch_and_add(&new_frontier->count, partial_frontier.size());
        #pragma omp parallel for                                                        
        for (int i = 0; i < partial_frontier.size(); ++i) {            
            new_frontier->vertices[i + index] = partial_frontier[i];
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for                                                        
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);
        //barrier();
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

bool bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* flags,
    int* distances, 
    int next_dist)
{
    // std::cout << "bottom UP" << std::endl;
    int cur_dist = next_dist;//distances[frontier->vertices[0]] + 1;
    bool frontier_left = false;

    // for each vertex v in graph:
    #pragma omp parallel
    {
        vertex_set partial_frontier;
        vertex_set_init(&partial_frontier, g->num_nodes);
        

        // if v has not been visited 
        #pragma omp for
        for (int v = 0; v < g->num_nodes; ++v) {
            if (distances[v] != NOT_VISITED_MARKER) continue;

            // check if v shares an incoming edge with a vertex u on the frontier
            bool shares_edge = false;
            int start_edge = g->incoming_starts[v];
            int end_edge = (v == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[v + 1];
            
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                if (flags[g->incoming_edges[neighbor]] == 1) {
                    shares_edge = true;
                    break;
                }
            }
            
            if (shares_edge) {
                // add vertex v to frontier
                partial_frontier.vertices[partial_frontier.count++] = v;
                distances[v] = cur_dist;
                // #pragma omp critical
                // std::cout << v << " is dist " << cur_dist << std::endl;
                frontier_left = true;
            }
        }

        #pragma omp barrier

        // int index = __sync_fetch_and_add(&new_frontier->count, partial_frontier.count);
        // memcpy(new_frontier->vertices + index, (partial_frontier.vertices), sizeof(int)*partial_frontier.count);

        for (int i = 0; i < partial_frontier.count; i++) {
            flags[partial_frontier.vertices[i]] = 1;
            // #pragma omp critical
            // std::cout << "adding " << partial_frontier.vertices[i] << " to the frontier." << std::endl; 
        }
    }

    return frontier_left;
}


void bfs_bottom_up(Graph graph, solution* sol)
{
    // std::cout << "edge to node ratio: " << graph->num_edges / (float) graph->num_nodes << std::endl;

    vertex_set list1;
    vertex_set list2;
    vertex_set list3;
    vertex_set list4;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set_init(&list3, graph->num_nodes);
    vertex_set_init(&list4, graph->num_nodes);

    vertex_set* frontier      = &list1;
    vertex_set* new_frontier  = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int* flags = (int *)calloc(graph->num_nodes, sizeof(int)); //new int[graph->num_nodes];
    flags[ROOT_NODE_ID] = 1;

    bool work_to_do = true;
    int next_dist = 1;

    while (work_to_do) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        work_to_do = bottom_up_step(graph, frontier, new_frontier, flags, sol->distances, next_dist);
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        next_dist++;
    }

    free(flags);
}


void bfs_hybrid(Graph graph, solution* sol)
{
    float ratio = graph->num_edges / (float)graph->num_nodes;
    // std::cout << "Ratio: " << ratio << std::endl;
    
    // start with top down bfs
    // keep aggregate sum of # nodes visited (in frontier)
    // at some threshold, swap to bottom up bfs

    vertex_set list1;
    vertex_set list2;
    vertex_set list3;
    vertex_set list4;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier      = &list1;
    vertex_set* new_frontier  = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for                                                   
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }
    sol->distances[ROOT_NODE_ID] = 0;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;

    int* flags = (int *)calloc(graph->num_nodes, sizeof(int)); 
    flags[0] = 1;

    int nodes_visited = 1;
    bool has_run_bottom_up = false;
    int next_dist = 1;

    double threshold = 0.5;
    if (ratio > 30) threshold = 0.80;
    bool work_to_do = true;

    while (frontier->count != 0 || work_to_do) {

        vertex_set_clear(new_frontier);
        

        if (nodes_visited < (int)(graph->num_nodes*threshold)) {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        } else {
            if (!has_run_bottom_up) {
                #pragma omp parallel for
                for (int i = 0; i < frontier->count; ++i) {
                    flags[frontier->vertices[i]] = 1;
                }
            }

            work_to_do = bottom_up_step(graph, frontier, new_frontier, flags, sol->distances, next_dist);
        }


        nodes_visited += new_frontier->count;

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        next_dist++;
    }

    free(flags);

}
