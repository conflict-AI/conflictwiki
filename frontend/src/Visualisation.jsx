import { ForceGraph2D } from "react-force-graph";
import React from "react";
import { sampleSize } from "lodash";

let node_list = require("./network/nodes.json");
let edge_list = require("./network/edges.json");

function transformNodes(nodes) {
    return nodes.map((node) => ({
	type: "NODE",
	id: String(node.entity_id),
	name: node.entity_name,
	url: node.url,
	num_conflicts: node.num_conflicts,
	language: node.language,
	ideology: node.ideology,
    }));
}

function transformEdges(edges) {
    return edges.map((edge) => ({
	type: "EDGE",
	source: String(edge.entity_id_1),
	target: String(edge.entity_id_2),
	conflict_attributes: edge.conflict_attributes,
    }));
}

function formatNodes(node) {
    return "black";
}
// edge colouring represents relationship. green --allies, red --enemies

function formatLinkColour(edge) {
    if (edge.conflict_attributes.label_discrete == 1) {
	return "#91cfa2";
    }

    return "#ff6666";
}

// edge line width represents the number of conflicts.
function formatLinkWidth(link) {
    var min = 1;
    var max = 20;

    // okay that is starting to look better.
    return link.conflict_attributes.n_conflicts * 0.5;
    //   return 0.1;
}

function formatLinkLabel(link) {
    if (link.conflict_attributes.conflict_names.length > 1) {
	var set = new Set(link.conflict_attributes.conflict_names);

	var mapped = Array.from(set);

	return mapped.length + " Conflicts";
    }

    return link.conflict_attributes.conflict_names[0];
}
// using react Memo to prevent the rerendering of the graph

class VisualisationGraph extends React.Component {
    constructor(props) {
	super(props);

	this.nodes = transformNodes(node_list);
	this.edges = transformEdges(edge_list);

	this.lookup = {};
	for (let node of this.nodes) {
	    this.lookup[node.id] = node;
	}

	this.state = {
	    size: this.props.size,
	    resample: this.props.resample,
	};
    }

    sampleGraph() {
	console.log("SAMPLING GRAPH");

	var data = {
	    links: sampleSize(this.edges, 400),
	};

	var nodes = new Set();

	for (let link of data.links) {
	    nodes.add(link.source);
	    nodes.add(link.target);
	}

	// WARNING, really strange bug
	// set does not filter as we want
	// some really janky logic is thus implemented to add difference
	// very bad code.

	// EDIT -> I think there is some kind of mutation where objects are being added to
	// the set where there should just be ints.
	// cant figure out how though (i think related to the visualisation library)
	// THe sample above should

	var filtered_nodes = [];
	for (let node of this.nodes) {
	    if (nodes.has(node.id)) {
		filtered_nodes.push(node);
	    }
	}

	let filtered_set = new Set(filtered_nodes.map((item) => item.id));
	let difference = new Set([...nodes].filter((x) => !filtered_set.has(x)));

	data.nodes = filtered_nodes;
	data.nodes.push(...difference);

	return data;
    }

    shouldComponentUpdate(nextProps, nextState) {
	console.log("attempting to update");
	// manually calculate (was having problems with app triggering rerender).
	if (this.props.size != nextProps.size) {
	    return true;
	}
	if (this.props.resample != nextProps.resample) {
	    console.log("RESAMPLEING");

	    return true;
	}

	return false;
    }

    render() {
	var data = this.sampleGraph();

	return (
	    <ForceGraph2D
		height={this.state.size.height}
		width={this.state.size.width}
		graphData={data}
		nodeColor={formatNodes}
		linkColor={formatLinkColour}
		linkWidth={formatLinkWidth}
		onNodeClick={this.props.setClickedElement}
		onLinkClick={this.props.setClickedElement}
		onLinkHover={this.props.updateLink}
		linkLabel={formatLinkLabel}
		nodeRelSize={7}
	    />
	);
    }
}
export default VisualisationGraph;
