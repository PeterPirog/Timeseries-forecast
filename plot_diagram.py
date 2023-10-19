import xml.etree.ElementTree as ET

root = ET.Element('graphml', xmlns="http://graphml.graphdrawing.org/xmlns",
                  xmlns_xsi="http://www.w3.org/2001/XMLSchema-instance",
                  xsi_schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd")

graph = ET.SubElement(root, 'graph', edgedefault="directed")

nodes = [
    "Univariate Models", "AR", "ARMA", "ARIMA", "SARIMA", "Holt", "Holt-Winters", "STL", "MSTL",
    "Multivariate Models", "VAR", "VMA", "VARIMA", "Exogenous",
    "Non-linear Models", "TAR", "STAR", "Switching Models",
    "Neural Networks", "RNN", "LSTM", "GRU", "FNN",
    "Gradient Trees", "XGBoost", "LightGBM", "CatBoost"
]

edges = [
    ("Univariate Models", "AR"),
    ("Univariate Models", "ARMA"),
    ("Univariate Models", "ARIMA"),
    ("Univariate Models", "SARIMA"),
    ("Univariate Models", "Holt"),
    ("Univariate Models", "Holt-Winters"),
    ("Univariate Models", "STL"),
    ("Univariate Models", "MSTL"),

    ("Multivariate Models", "VAR"),
    ("Multivariate Models", "VMA"),
    ("Multivariate Models", "VARIMA"),
    ("Multivariate Models", "Exogenous"),

    ("Non-linear Models", "TAR"),
    ("Non-linear Models", "STAR"),
    ("Non-linear Models", "Switching Models"),

    ("Neural Networks", "RNN"),
    ("Neural Networks", "LSTM"),
    ("Neural Networks", "GRU"),
    ("Neural Networks", "FNN"),

    ("Gradient Trees", "XGBoost"),
    ("Gradient Trees", "LightGBM"),
    ("Gradient Trees", "CatBoost"),
]

for n in nodes:
    node_element = ET.SubElement(graph, 'node', id=n, label=n)

for (src, dest) in edges:
    edge_element = ET.SubElement(graph, 'edge', source=src, target=dest)

tree = ET.ElementTree(root)
tree.write('timeseries_taxonomy.graphml', encoding="utf-8", xml_declaration=True)
