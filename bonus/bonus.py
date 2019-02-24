import networkx as nx

def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined in bonus.md.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    ###TODO

    neighbor = set(graph.neighbors(node))
    nodes = [x for x in graph.nodes() if x not in neighbor and x != node]

    scores = []
    for ni in nodes:
        x = 0.
        deg = 0.
        for n1 in neighbor:
            deg = graph.degree(n1)
            x = x + deg
        x = (1/x)

        neighbor2 = set(graph.neighbors(ni))
        y = 0.
        deg = 0.
        for n2 in neighbor2:
            deg = graph.degree(n2)
            y = y + deg           
        y = (1/y)
        
        intersection = set(neighbor & neighbor2)
        z = 0.
        deg = 0.
        for n in intersection:
            deg = graph.degree(n)
            z = z + (1/deg)
        
        score = ((node, ni), 1. * (z/(x + y)) )
        scores.append(score)

    return sorted(scores, key = lambda x: (-x[1], x[0][1]))

#A sample graph is created to test the above function.
#def example_graph():
#    g = nx.Graph()
#    g.add_edges_from([('A', 'G'), ('A', 'C'), ('C', 'H'), ('B', 'C'), ('H', 'G'), ('B', 'D'), ('D', 'E'), ('D', 'F')])
#    return g

#Main function is written to verify if this program works.
#if __name__ == '__main__':
#    g = example_graph()
#    print(jaccard_wt(g, 'B'))