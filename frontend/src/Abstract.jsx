function Abstract() {
  return (
    <div>
      <br />

      <h3>Abstract</h3>
      <p>Mitigation of militarised conflict requires an understanding of conflict causes and the relationships of warring entities. Peace and Conflict Studies seeks to build this understanding by considering differing ideologies of entity pairs (dyadic theories) and multi-lateral relationship structures (systemic theories). However, dyadic and systemic theories are not only entangled, but hard to operationalise and validate. The available event-level, tabular data is limiting quantitative models to mostly theory-free prediction. To address this issue, we propose combining rich textual and topological, conflict-level data with an expressive graph neural network. We publish the ConflictWiki dataset, a large graph extracted from Wikipedia where nodes represent entities connected by labelled edges representing “ally” or “enemy”-relationships in conflict. Node and edge features are given by Lonformer document embeddings of entity and conflict articles. This allows casting the problem as an edge classification task holding back certain textual or topological information depending on the dyadic or systemic setting. We obtain slightly stronger results in the systemic setting and find that the articles of entities are more informative for the ally--enemy classification task than the conflict article itself. Articles of allies are semantically more similar than those of enemies and certain sections (e.g. religion, politics) are more conflict-indicative overall.</p>
    </div>
  );
};

export default Abstract;
