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
    >
	{(close) => (
	    <Modal.Dialog scrollable>
		<Modal.Header closeButton onHide={close}>
		    <Modal.Title>About</Modal.Title>
		</Modal.Header>

		<Modal.Body>
		    <p>
			Please find the link to the paper{" "}
			<a href="https://arxiv.org/abs/2109.12860">here</a>.
	  </p>

		    <p>
			Github repository with all the code, data and tutorials on how to
			use the data can be found
	    <a href=" https://github.com/conflict-ai/conflictwiki"> here </a>
		    </p>
		    <p>
			Please direct any questions to{" "}
			<a href="mailto:niklas.stoehr@inf.ethz.ch">
			    niklas.stoehr@inf.ethz.ch
	    </a>
		    </p>
		    <h4>How To Cite</h4>
		    <p>
			{`@article{

stoehr2021classifying,
title={Classifying Dyads for Militarized Conflict Analysis},
author={Niklas Stoehr, Lucas Torroba Hennigen, Samin Ahbab, Robert West and Ryan Cotterell},
booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
year={2021},
url={https://conflict-ai.github.io/conflictwiki/},
}
			    `}
		    </p>
		</Modal.Body>
	    </Modal.Dialog>
	)}
    </Popup>
);
