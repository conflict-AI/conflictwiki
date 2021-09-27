import React, { useEffect, useState } from "react";
import Abstract from "./Abstract";
import LinkChoice from "./edgeLinkChoice";


export default function InfoSection({ element, update }) {
  // to display information of clicked nodes.

  // if (element == {}) return <Abstract />;
  // if (element === null) return null;

  if (element.type == "NODE") {
    var google = `<iframe width='100%' height='100%' scrolling='yes' src='${element.url}' sandbox='allow-modals allow-forms allow-popups allow-scripts allow-same-origin style="-webkit-transform:scale(0.5);-moz-transform-scale(0.5);"'></iframe>`;

    return <div
      style={{ height: "100%", width: "100%" }}
      dangerouslySetInnerHTML={{ __html: google ? google : "" }}
    />;
  }

  if (element.type == "EDGE") {
    return <LinkChoice link={element} updateLink={update} />;
  }

  if (element.type == "EDGE_LINK") {
    console.log(element);
    var google = `<iframe width='100%' height='100%' scrolling='yes' src='${element.url}' sandbox='allow-modals allow-forms allow-popups allow-scripts allow-same-origin style="-webkit-transform:scale(0.5);-moz-transform-scale(0.5);"'></iframe>`;

    return <div
      style={{ height: "100%", width: "100%" }}
      dangerouslySetInnerHTML={{ __html: google ? google : "" }}
    />;
  }




  return (<div style={{ height: "100%", width: "100%" }}>

    <div>
      <h1 className="text-justify">
	Click on a node or edge to display its wikipedia content
	  </h1>
      <Abstract />
    </div>


  </div>
  );
}
