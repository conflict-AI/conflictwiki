import React from "react";
import { isEmpty } from "lodash";

import Card from "react-bootstrap/Card";

// function transformConflictNames(names) {
//   let set = new Set(names);
//   return Array.from(set).map(x => <p>{x}</p>);
// }

// TODO: slice the lists to max 4
function transformList(list) {
  var set = new Set(list);

  var mapped = Array.from(set).map((element) => <p>{element}</p>);

  return mapped;
}

function GraphInfo({ node, link }) {
  return (
    <div style={{ position: "absolute", zindex: "100" }}>
      <Card
	style={{
	  fontSize: 10,
	  zIndex: "100",
	  position: "absolute",
	  width: "14rem",
	}}
      >
	<Card.Body>
	  <b>EDGE INFORMATION</b>
	  <br />
	  <br />

	  {!isEmpty(link) && (
	    <div>
	      {link.source.name + " <-> " + link.target.name}
	      <br />
	      <br />
	      <b>{link.conflict_attributes.label}</b>
	      <br />
	      <br />
	      {transformList(link.conflict_attributes.conflict_names)}
	    </div>
	  )}
	</Card.Body>
      </Card>
      <Card
	style={{
	  fontSize: 10,
	  position: "absolute",
	  left: "250px",
	  zIndex: "100",
	  width: "12rem",
	}}
      >
	<Card.Body>
	  <b>NODE INFORMATION</b>
	  <br />
	  <b>{node.name}</b>

	  <br />
	  {node.num_conflicts != undefined
	    ? node.num_conflicts + " Conflicts"
	    : ""}
	  <br />
	</Card.Body>
      </Card>
    </div>
  );
}

export default GraphInfo;
