import React from "react";
import Popup from "reactjs-popup";
import Button from "react-bootstrap/Button";

import Modal from "react-bootstrap/Modal";


export default () => (
  <Popup
    trigger={
      <Button
	style={{
	  position: "absolute",
	  top: "0px",
	  right: "215px",
	  zIndex: "100",
	}}
	variant="secondary "
      >
	About
      </Button>
    }
    position="right center"
    modal
  >{close =>
    <Modal.Dialog scrollable>
      <Modal.Header closeButton onHide={close}>
	<Modal.Title>About</Modal.Title>
      </Modal.Header>

      <Modal.Body>

	<p>This work is currently under submission and therefore anonymised.</p>
	<h4>Ethics Statement</h4>
	<p>Our interdisciplinary paper bridges the academic disciplines of Natural Language Processing and Conflict and Peace Studies. The classification of armed conflict and its assessment of any kind is highly sensitive, subjective and context-dependent. Therefore, we have decided to limit our theory-driven classifications to historic data. Many supra-national and non-governmental institutions rely on computational models for conflict classification for peaceful, humanitarian purposes such as conflict mitigating and prevention. These models play an important role in the compilation of comprehensive state reports, war journalism, assessment of war crime, provision of humanitarian aid and international, jurisprudential decision-making. We would like to contribute to a transparent scholarly debate, rather than leaving ethically complex questions to secretive enterprises or governmental institutions. Even more so, since conflict classification models are already deployed on a daily basis in real-world applications, despite little scrutiny. Assessment of conflict should strive for more diverse, context-dependent perspectives accounting for psychological, socio-economic damage and more subtle, but no less important factors. We are aware of the risk of bias within <a href="https://en.wikipedia.org/wiki/Main_Page">Wikipedia</a>. Our textual data is limited to English language only, but does not violate privacy rights by disclosing identifiable individuals. The countries analysed in our study are selected solely based on data availability. Training our models on empirical data, results are at risk of replicating inherent data bias.</p>
      </Modal.Body>

    </Modal.Dialog>

    }
  </Popup>
);
