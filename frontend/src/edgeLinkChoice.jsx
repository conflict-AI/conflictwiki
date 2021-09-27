import React from "react";
import ListGroup from "react-bootstrap/ListGroup";

const formatLink = (link) => {
  return "https://en.wikipedia.org/wiki/" + link.replace(" ", "_");
};


export default function LinkChoice({ link, updateLink }) {



  if (link == undefined) {
    return "UNDEFINED";
  }


  var set = new Set(link.conflict_attributes.conflict_names);
  if (set.size == 1) {
    var setIter = set.values();

    updateLink({ type: "EDGE_LINK", url: formatLink(setIter.next().value) });

  }
  var mapped = Array.from(set).map((element) => <ListGroup.Item action onClick={() => updateLink({
    type: "EDGE_LINK", url: formatLink(element)
  })}>{element}</ListGroup.Item>);

  return <div>
    <h5 className="text-justify">
      Click on a link to display its wikipedia content
	  </h5>

    <br />
    <ListGroup>{mapped}</ListGroup> </div>;
}
