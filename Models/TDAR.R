library(ggplot2)
library(igraph)
library(networkD3)
library(parallel)
library(foreach)
library(doParallel)
library(tidyverse)

source('~/Desktop/TDA-R/MapperAlgo/R/EdgeVertices.R')
source('~/Desktop/TDA-R/MapperAlgo/R/ConvertLevelsets.R')
source('~/Desktop/TDA-R/MapperAlgo/R/Cover.R')
source('~/Desktop/TDA-R/MapperAlgo/R/Cluster.R')
source('~/Desktop/TDA-R/MapperAlgo/R/SimplicialComplex.R')
source('~/Desktop/TDA-R/MapperAlgo/R/MapperAlgo.R')
source('~/Desktop/TDA-R/MapperAlgo/R/Plotter.R')
source('~/Desktop/TDA-R/MapperAlgo/R/GridSearch.R')

filter_full = read_csv("./ComputedData/ForModel/filtered_data.csv") %>% select(-c(pc4, pc5))

filter_full

time_taken <- system.time({
  Mapper <- MapperAlgo(
    filter_values = filter_full,
    percent_overlap = 30,
    methods = "dbscan",
    method_params = list(eps = 0.3, minPts = 1),
    # methods = "hierarchical",
    # method_params = list(num_bins_when_clustering = 10, method = 'ward.D2'),
    # methods = "kmeans",
    # method_params = list(max_kmeans_clusters = 2),
    # methods = "pam",
    # method_params = list(num_clusters = 2),
    cover_type = 'stride',
    # intervals = 4,
    interval_width = 0.7,
    num_cores = 10
  )
})

MapperPlotter(Mapper, filter_full$centrality, filter_full, type = "forceNetwork")

