import mlflow

def calculate_sum(x,y):
    return x+y



if __name__ == '__main__':
    #start the server of the ml flow 
    with mlflow.start_run():
        x,y =180,820
        z= calculate_sum(x,y)
        #track the experiment with the ml flow
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("z",z)
    #print("sum",S)

# how i track this particular experiment 