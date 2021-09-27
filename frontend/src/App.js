import "./App.css";

import Button from "react-bootstrap/Button";
import Jumbotron from "react-bootstrap/Jumbotron";
import Container from "react-bootstrap/Container";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import cloneDeep from "lodash";
import React, { useCallback, useState } from "react";

import cambridge from "./logos/cambridge.png";
import zurich from "./logos/zurich.png";
import epfl from "./logos/epfl.png";
import mit from "./logos/mit.png";

import VisualisationGraph from "./Visualisation";

import InfoSection from "./infoColumn";
import GraphInfo from "./GraphInfo";
import InfoPopup from "./popup";

import { SizeMe } from "react-sizeme";
import { isMobile } from "react-device-detect";



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
    if (isMobile) {
      return (<div> This app is designed only for desktop. Please view on desktop.</div>);
    }
    return (
      <Container fluid>
	<Row>
	  <Col xs={8} sm={8} md={8} lg={8} xl={8}>
	    <h2> <b>ConflictWiki</b>	      Classifying Dyads for Militarized Conflict Analysis </h2>
	    <h5>Empirical Methods in Natural Language Processing (EMNLP) 2021</h5>

	    <a href="https://niklas-stoehr.com">Niklas Stoehr  </a> &nbsp; &nbsp;
	  <a href="https://ltorroba.github.io">Lucas Torroba Hennigen </a> &nbsp; &nbsp;
	  <a href="https://www.linkedin.com/in/samin-ahbab-4ba45a43/">Samin Ahbab   </a> &nbsp; &nbsp;
	    <a href="https://dlab.epfl.ch/people/west/">   Robert West </a> &nbsp; &nbsp;
	    <a href="https://rycolab.io/authors/ryan/"> Ryan Cotterell </a>
	    <br />
	    <br />
	  </Col>
	  <Col xs={4} sm={4} md={4} lg={4} xl={4}>
	    <br />
	    <br />
	    <img src={zurich} style={{ width: "100px", height: "10" }} />
	    <img src={mit} style={{ width: "100px", height: "10" }} />
	    <img src={epfl} style={{ width: "70px", height: "17px" }} />
	    <img src={cambridge} style={{ width: "100px", height: "10" }} />

	  </Col>
	</Row>

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
