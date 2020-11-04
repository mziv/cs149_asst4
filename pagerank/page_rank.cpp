#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>
#include <iostream>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{
  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  #pragma omp parallel for                                                        
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }

  // TODO: Lock the vector or flag this?
  std::vector<Vertex> no_outgoing_edges;
  for (int i = 0; i < numNodes; ++i) {
    if (outgoing_size(g, i) == 0) {
      no_outgoing_edges.push_back(i);
    }
  }

  int num_no_outgoing = no_outgoing_edges.size();
  double *score_new = new double[numNodes];
    
  #pragma omp parallel for                                                        
  for (int i = 0; i < numNodes; i++){
    score_new[i] = 0.0;
  }

  bool converged = false;
  double global_diff = 0.0;

 while (!converged) {
    // loop over incoming edges and store sum in score_new
    double sum_no_outgoing = 0.0;

    // TODO: keep serial?
    for (int k = 0; k < num_no_outgoing; ++k) { // k for Kayvon and Kunle <3
      sum_no_outgoing += (damping * solution[no_outgoing_edges[k]]) / numNodes;
    }

    #pragma omp parallel for                                                        
    for (int i = 0; i < numNodes; ++i) {
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      for (const Vertex* j = start; j != end; ++j) {
        score_new[i] += (solution[*j] / outgoing_size(g, *j));
      }
      score_new[i] = (damping * score_new[i]) + (1.0-damping) / numNodes;
      score_new[i] += sum_no_outgoing;
    }

    for (int l = 0; l < numNodes; ++l) {
      global_diff += std::abs(score_new[l] - solution[l]);
    }

    converged = (global_diff < convergence);
    global_diff = 0.0;

    #pragma omp parallel for                                                        
    for (int i = 0; i < numNodes; i++) {
      solution[i] = score_new[i];
      score_new[i] = 0.0;
    }
  }
  delete[] score_new;

  
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
