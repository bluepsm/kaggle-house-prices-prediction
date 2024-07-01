import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn.inspection import DecisionBoundaryDisplay


def plot_scatter(x, y, x_label, y_label, classes, classes_name):
    fig = plt.figure(1)
    ax = fig.add_subplot()
    scatter = ax.scatter(x, y, c=classes)
    ax.set(xlabel=x_label, ylabel=y_label)
    _ = ax.legend(scatter.legend_elements()[0], classes_name, loc='lower right', title='Classes')
    plt.show()


def plot_scatter_3d(x, y, z, classes, classes_name, x_label='X', y_label='Y', z_label='Z'):
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
    scatter = ax.scatter(x, y, z, c=classes, s=40)
    ax.set_xlabel(x_label)
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel(y_label)
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel(z_label)
    ax.zaxis.set_ticklabels([])
    _ = ax.legend(scatter.legend_elements()[0], classes_name, loc='lower right', title='Classes')
    plt.show()


# Plot with Logistics Regression
def plot_scatter_logreg(logreg_model, features, x_label, y_label, labels):
    fig = plt.figure(3)
    ax = fig.add_subplot()
    DecisionBoundaryDisplay.from_estimator(
        logreg_model,
        features,
        cmap=plt.cm.Paired,
        ax=ax,
        response_method='predict',
        plot_method='pcolormesh',
        shading='auto',
        xlabel=x_label,
        ylabel=y_label,
        eps=0.5
    )
    plt.scatter(features[:, 0], features[:, 1], c=labels, edgecolors='k', cmap=plt.cm.Paired)
    plt.xticks(())
    plt.yticks(())
    plt.show()


