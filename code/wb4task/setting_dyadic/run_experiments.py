from wb0configs import configs
from wb4task.setting_transductive.trans_batched_dataload_class import load_data
from wb4task.setting_transductive.trans_batched_train_class import Trainer
from wb4task.setting_dyadic.no_gnn import Model, Model_Trainer



def run_experiments():

    for data_seed in list(range(10)):

        dl, val_test_dl, graph, label_info, wikinetworkdata = load_data(config, batch_size = 512, n_val= 0.3, n_test= 0.1, neighborhood_steps= 2, random_node_frac = 0.0, random_edge_frac = 0.0, random_label_frac = 0.0, random_seed=data_seed)

        model = Model(node_feature_dim = 500, node_reducer_out = 512, dot_product_dim = 96, include_edge_features = True)

        trainer = Model_Trainer(wikinetworkdata, label_info)
        model = trainer.train_model(dl, model, graph, n_epochs=2, early_stop=3, lr = 0.001, weight_decay=1e-5)
        trainer.evaluate_model(model, val_test_dl)



def grid_search():

    gnn_in_features = [32, 64, 128, 256, 512]
    gnn_out_features = [6, 12, 24, 48, 96]

    for in_feat in gnn_in_features:
        for out_feat in gnn_out_features:
            print("in_feat:", in_feat, "out_feat:", out_feat)

            dl, val_test_dl, graph, label_info, wikinetworkdata = load_data(config, batch_size=512, n_val=0.3,
                                                                            n_test=0.1, neighborhood_steps=1,
                                                                            random_node_frac=0.0, random_edge_frac=0.0,
                                                                            random_label_frac=0.0)

            model = Model(node_feature_dim=500, node_reducer_out=in_feat, dot_product_dim=out_feat,
                          include_edge_features=True)

            trainer = Model_Trainer(wikinetworkdata, label_info)
            model = trainer.train_model(dl, model, graph, n_epochs=30, early_stop=3, lr=0.001, weight_decay=1e-5)
            trainer.evaluate_model(model, val_test_dl)

if __name__ == "__main__":

    config = configs.ConfigBase()
    #grid_search()
    run_experiments()
