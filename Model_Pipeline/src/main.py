import omegaconf
from datamodule import YelpDataset
from models import Model
from trainer import Trainer
from transformers import get_linear_schedule_with_warmup
import torch


# Training pipeline
class train_pipeline(object):

    # Function: read config files and create model save dir
    def __init__(self, config):
        # load config files
        self.config = omegaconf.OmegaConf.load(config)

    # Function: dataset load
    def data_loading(self):
        # read the params
        data_params = {
                            "trainset_path": self.config.datamodule.trainset_path,
                            "testset_path": self.config.datamodule.testset_path,
                            "feature_col": self.config.datamodule.feature_col,
                            "label_col": self.config.datamodule.label_col,
                            "max_length": self.config.datamodule.max_length,
                            "batch_size": self.config.datamodule.batch_size,
                            "test_size": self.config.datamodule.test_size,
                            "stratify_col": self.config.datamodule.stratify_col,
                            "tokenizer_name": self.config.tokenizer_name,
                            "seed": self.config.seed,
                            "device": self.config.device,
                        }
        yelp_dataset = YelpDataset(**data_params)
        # prepared the data loader for model training
        self.train_loader = yelp_dataset.train_dataloader()
        self.valid_loader = yelp_dataset.valid_dataloader()

    # Function: model load
    def model_loading(self):
        model_params = {
                            "_num_labels": self.config.model.num_labels,
                            "_model_name": self.config.pretrained_model,
                            "device": self.config.device,
                            "unfreeze_layers": self.config.model.unfreeze_layers,
                            "dropout": self.config.model.dropout,
                            "activation": self.config.model.activation,
                            "hidden_size": self.config.model.hidden_size,
                            "token": self.config.tokenizer_name,
                            "layer_sizes": self.config.model.layer_sizes,
                        }
        self.model = Model(**model_params)

    # Function: train model
    def train_model(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.trainer.lr,
            weight_decay=self.config.trainer.weight_decay,
        )

        num_training_steps = self.config.trainer.num_epochs * len(self.train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * self.config.trainer.warm_up_step),
            num_training_steps=num_training_steps,
        )

        trainer_params = {
            "model": self.model,
            "train_loader": self.train_loader,
            "valid_loader": self.valid_loader,
            "config": self.config,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "early_stop": self.config.trainer.early_stop,
            "device": self.config.device,
        }

        trainer = Trainer(**trainer_params)
        trainer.fit(num_epochs=self.config.trainer.num_epochs)


    # Function: Training Pipeline
    def start_pipeline(self):
        self.data_loading()
        self.model_loading()
        self.train_model()

# === Main access
if __name__ == "__main__":
    # select the pre-trained model for training
    pre_trained_model = "uncased" # [deberta, uncased, t5]
    # start training
    train_model = train_pipeline("../config/config_"+pre_trained_model+".yaml")
    train_model.start_pipeline()