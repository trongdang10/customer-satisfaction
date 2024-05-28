from zenml.client import Client

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    #run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path = "C:/Users/viett/customer_satisfaction/data/olist_customers_dataset.csv")

