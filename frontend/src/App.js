import "./App.css";

import Button from "react-bootstrap/Button";
import Jumbotron from "react-bootstrap/Jumbotron";
import Container from "react-bootstrap/Container";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import cloneDeep from "lodash";
import React, { useCallback, useState } from "react";

import VisualisationGraph from "./Visualisation";

import InfoSection from "./infoColumn";
import GraphInfo from "./GraphInfo";
import InfoPopup from "./popup";

import { SizeMe } from "react-sizeme";

// function handleClick() {
//   console.log("Button Clicked!");
// }

class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      clickedElement: {},
      hoveredLink: {},
      resample: 1,
    };

    this.setClickedElement = this.setClickedElement.bind(this);
    this.setHoveredLink = this.setHoveredLink.bind(this);
    this.triggerResample = this.triggerResample.bind(this);
  }

  setClickedElement(element) {
    this.setState({ clickedElement: element });
  }

  setHoveredLink(link) {
    console.log(link);
    this.setState({ hoveredLink: link });
  }

  triggerResample() {
    this.setState({ resample: this.state.resample + 1 });
  }

  render() {
    return (
      <Container fluid>
	<div>
	  <h1> ConflictWiki </h1>
	  <h3>
	    Classifying Dyads for Militarized Conflict Analysis
	  </h3>
	  <h5>Published in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)
</h5>
	  <a href="https://niklas-stoehr.com">Niklas Stoehr  </a> &nbsp; &nbsp;
	  <a href="https://ltorroba.github.io">Lucas Torroba Hennigen </a> &nbsp; &nbsp;
	  <a href="https://www.linkedin.com/in/samin-ahbab-4ba45a43/">Samin Ahbab   </a> &nbsp; &nbsp;
	  <a href="https://rycolab.io/authors/ryan/">   Robert West </a>
	</div>

	<Row className="vh-100 d-flex">
	  <Col xs={3} sm={7} className="vh-100 d-flex">
	    <SizeMe monitorWidth monitorHeight>
	      {({ size }) => (
		<div style={{ width: "100%", height: "100%" }}>
		  <Button
		    style={{
		      position: "absolute",
		      top: "0px",
		      right: "0px",
		      zIndex: "100",
		      marginLeft: "auto",
		    }}
		    variant="secondary"
		    onClick={() => this.triggerResample()}
		  >
		    Click To Resample Graph
		  </Button>
		  <InfoPopup />
		  <GraphInfo
		    node={this.state.clickedElement}
		    link={this.state.hoveredLink}
		  />

		  <VisualisationGraph
		    setClickedElement={this.setClickedElement}
		    updateLink={this.setHoveredLink}
		    size={size}
		    resample={this.state.resample}
		  />
		</div>
	      )}
	    </SizeMe>
	  </Col>
	  <Col
	    xs={9}
	    sm={5}
	    className="vh-100 d-flex" /* style={{ height: "100px" }} */
	  >
	    <InfoSection
	      element={this.state.clickedElement}
	      update={this.setClickedElement}
	    />
	  </Col>
	</Row>
      </Container >
    );
  }
}

export default App;
