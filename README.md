# ConflictWiki

This repository houses the code and dataset for the paper *Text or Topology? Classifying Ally-Enemy Pairs in Militarised Conflict*

Niklas Stoehr, Lucas Torroba Hennigen, Samin Ahbab, Robert West, Ryan Cotterell

ETH Zurich, MIT, EPFL, University of Cambridge

# ConflictWiki Data v0.1

We introduce the ConflictWiki dataset, a large collection of [Wikipedia articles on armed conflict](https://en.wikipedia.org/wiki/Category:Conflicts) featuring full text accompanied by pre-computed meaningful Longformer representations of the text on the document- and section-level.
Each conflict is annotated with starting and end date, location, casualty counts, group strengths and conflict outcome.
If provided on Wikipedia, entity articles feature auxiliary tabular information on languages, religion, ISO2 code and ideology.

For examples on how to load and visualise the data please see here:

[loading all data](tutorials/ConflictWiki%20Data.ipynb) <br>
[visualising the data as network](tutorials/ConflictWiki%20Network.ipynb)

![alt text](paper-code/tutorials/pictures/overview.png "overview")


## Data Directory Structure

The data directory contains three subdirectory, `conflict`, `entity` and `mappings`.
`conflict` contains conflict-related data and information.
`entity` contains entity-related data and information.
`mappings` contains the conflict-entity and entity-entity relationships.
If there are multiple files in a directory, they all contain the same content.
We provide most data in multiple data formats (json, csv, pickle).
The contents of each subdirectory is explained in the following.

```
|   data
	|   conflict
		|   data
				| embedding
				| text
		|   info
	|   entity
		|   data
				| embedding
				| text
		|   info
	|   mappings
		|   conflict_entity_id
		|   ally_enemy_pairs
		|   network
				|   aggr_edge_list
				|   node_list
```

### conflict

##### data

contains the retrieved conflict articles.

- `embedding` features the articles' Longformer representations as a pickled dictionary.
`conflict_id` is the conflict id; `entity_id` is the id of the entity as mentioned in the conflict article;
`embeddings` are the representations of the relevant entities in the article.
 We use the key `0` to represent the entire conflict article as a whole, irrespective of entities.

	```
	{conflict_id: entity_id: embeddings}
	```

- `text` is the raw full text.

	```
	{conflict_id: section_title: text}
	```

##### info

contains all information on the conflicts as extracted from the [Wikipedia militarized conflict template](https://en.wikipedia.org/wiki/Template:Infobox_military_conflict).

`conflict_id`, `conflict_name`, `place`,  `date`, `date_start`, `date_end`, `n_belligerents`, `n_entities`, `strength`, `strength_num`, `casualties`, `casualties_num`, `commander`, `commander_num`, `result` tags.<br>

date holds the date as a string, date_start and date_end are extracted from date and are datetime objects; <br>
n_belligerents, n_entities state the number of entities and belligerents involved in the conflict; <br>
strength and casualties estimate group strengths and losses per belligerent (dictionary keys are belligerent indeces); <br>
strength_num and casualties_num are numbers extracted from strength and casualties (dictionary keys are belligerent indeces); <br>
commander lists all military commanders per belligerent (dictionary keys are belligerent indeces); <br>
result is a brief textual summary of the outcome  <br>


### entity

##### data

contains the retrieved entity articles.

- `embedding` features the articles' Longformer representations as a pickled dictionary.
`entity_id` is the entity id; `section_title` is the title of the section;
`embeddings` are the representations of the sections in entity article

	```
	{entity_id: section_title: embeddings}
	```

- `text` is the raw full text.

	```
	{entity_id: section_title: text}
	```

##### info

contains all information on the entities as extracted from the [Wikipedia article info box](https://en.wikipedia.org/wiki/Mali).

`entity_id`, `entity_name`, `num_conflicts`, `iso`, `language`, `religion`, `ideology`<br>

num_conflicts gives the number of conflicts, the entity is involved in; <br>

### mappings

##### conflict_entity_id

Contains the ids of all conflicts and the ids of all involved entities partitioned into belligerents.
In the examplatory conflict below, we have three belligerents, of which the first two are formed by two entities each
and the third belligerent consists of only one entity

```
{conflict_id: [[entity_id, entity_id, entity_id, entity_id],[entity_id]]}
(1220919: [[1576797, 31717, 4887, 3434750], [40596311]])
```

##### ally_enemy_pairs

Contains the ids of entity pairs, a conflict id and the relationship of the entities in the given conflict as displayed below

```
((entity_id 1, entity_id 2), conflict_id, entity_relationship)
((27019, 103100), 22738, "enemies")
```

##### network

Contains the precomputed, aggregated network, where nodes are entities and edges are all conflicts between any two respective entities aggregated.
Aggregated dyad network; nodes represent entities, edges represent aggregated conflicts between entities, edge line width represents number of conflicts, edge colouring represents relationship

![alt text](tutorials/pictures/task_construction.png "task_construction")


###### aggr_edge_list

List of edges, each item is a triple (node_id 1, node_id 2, edge attributes)

```
(entity_id 1, entity_id 2, conflict attributes)
27019,	103100,	{'label': 'enemies', 'label_discrete': 0, 'label_continuous': -1, 'n_conflicts': 1, 'conflict_ids': [22738], 'conflict_names': ['Operation Enduring Freedom']}
```

###### node_list

List of nodes, each item is a tuple (node_id, node attributes)

```
(entity_id, entity attributes)
3343, {'name': 'Belgium'}
```
