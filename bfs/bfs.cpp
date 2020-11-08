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

void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    vertex_set* unvisited,
    vertex_set* new_unvisited,
    int* distances)
{
    // std::cout << "bottom UP" << std::endl;
    
    // Build a hash set of the frontier for easy inclusion test
    std::unordered_set<int> frontier_set;
    for (int i=0; i<frontier->count; i++) frontier_set.insert(frontier->vertices[i]);
    int cur_dist = distances[frontier->vertices[0]] + 1;

    // for each vertex v in graph:
    #pragma omp parallel
    {
        std::vector<int> partial_frontier;
        std::vector<int> partial_unvisited;

        // if v has not been visited 
        #pragma omp for
        for (int i = 0; i < unvisited->count; ++i) {
            int v = unvisited->vertices[i];            

            // check if v shares an incoming edge with a vertex u on the frontier
            bool shares_edge = false;
            int start_edge = g->incoming_starts[v];
            int end_edge = (v == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[v + 1];

            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                if (frontier_set.count(g->incoming_edges[neighbor]) > 0) {
                    shares_edge = true;
                    break;
                }
            }
            
            if (shares_edge) {
                // add vertex v to frontier
                partial_frontier.push_back(v);
                distances[v] = cur_dist;
            } else {
                // v is still unvisited
                partial_unvisited.push_back(v);
            }
        }

        #pragma omp barrier

        int index = __sync_fetch_and_add(&new_frontier->count, partial_frontier.size());
        #pragma omp parallel for                                                        
        for (int i = 0; i < partial_frontier.size(); ++i) {            
            new_frontier->vertices[i + index] = partial_frontier[i];
        }

        index = __sync_fetch_and_add(&new_unvisited->count, partial_unvisited.size());
        #pragma omp parallel for                                                        
        for (int i = 0; i < partial_unvisited.size(); ++i) {            
            new_unvisited->vertices[i + index] = partial_unvisited[i];
        }
    }
}


void bfs_bottom_up(Graph graph, solution* sol)
{
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
    vertex_set* unvisited     = &list3;
    vertex_set* new_unvisited = &list4;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for                                                        
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        unvisited->vertices[i] = i + 1; // WARNING: this assumes root node id is always 0
    }

    unvisited->count = graph->num_nodes - 1;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        vertex_set_clear(new_unvisited);

        bottom_up_step(graph, frontier, new_frontier, unvisited, new_unvisited, sol->distances);
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        tmp = unvisited;
        unvisited = new_unvisited;
        new_unvisited = tmp;
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
