from wb0configs import configs
from wb4task.dataload_class import load_data
from wb4task.models.C_task.model import Model
from wb4task.train_transductive.trans_batched_train_class import Trainer


class C_Trainer(Trainer):

    def __init__(self,wikidataset,entity_emb):
        super().__init__(wikidataset, entity_emb = entity_emb)

    def pass_data_to_model(self, model, x_1, x_2, c, e, y, l):
        y_hat = model(x_1, x_2)
        return y_hat


if __name__ == "__main__":

    config = configs.ConfigBase()
    train_dl, val_dl, test_dl, wikidataset = load_data(config, task_type= "conflict_entity_task", batch_size=32, n_val=0.4, n_test=0.05, limit_samples = None)

    model = Model()
    model = model.apply(model.init_weights)

    trainer = C_Trainer(wikidataset, entity_emb = "c")
    model = trainer.train_model(train_dl, val_dl, model, n_epochs=64, early_stop=12, lr = 0.001, weight_decay=1e-5)
    trainer.evaluate_model(model, test_dl)
