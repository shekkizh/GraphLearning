import graphlearning as gl

# Example of how to generate accuracy plots

# ssl_method_list = ['laplace', 'poisson']
# legend_list = ['KNN laplace', 'KNN poisson', 'NNK laplace', 'NNK poisson']
ssl_method_list = ['poisson']
legend_list = ['KNN poisson', 'NNK poisson']

# Select dataset to plot
dataset = "cifar_aet_k50" #"MNIST_vae_k10"
log_dirs = ["Results_KNN/", "Results_NNK/"]
directed_graph = True # False #
save_file = "accuracy.pdf"
num_classes = 10
gl.accuracy_plot(dataset, ssl_method_list, legend_list, num_classes, errorbars=True, testerror=False, loglog=False,
                 savefile=save_file, log_dirs=log_dirs, directed_graph=directed_graph)