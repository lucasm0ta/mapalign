import matplotlib.pyplot as plt
from mapalign.embed import DiffusionMapEmbedding
import numpy as np
from sklearn import manifold, datasets
import neighborhood as nb
from SelectFromCollection import SelectFromCollection

def plot_hit(values, samples, title):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.title(title)
    plt.xlabel('Neighbors')
    plt.ylabel('Percentage')
    ax = plt.subplot(111)
    ax.set_xlim(0, 1)
    labels = []
    maxim = 0
    for i, arr in enumerate(values):
        # print(i)
        dim = np.arange(0, arr.shape[0], 1)
        label, = plt.plot(arr, linestyle='--', label=str(samples[i]), alpha=0.7)
        labels.append(label)
        plt.xticks(dim)
        maxim = max([max(arr), maxim])
    plt.ylim(ymin=0, ymax=(maxim * 1.1))
    plt.legend(handles=labels, loc=4)
    # dim = np.arange(0, values.shape[0], 1)
    # plt.plot(values, linestyle='--', marker='^', color='b')
    plt.xticks(dim)
    plt.grid()
    plt.show()
    plt.close()

def plot_preservation(values, samples, title):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.title(title)
    plt.xlabel('Neighbors')
    plt.ylabel('Percentage')
    ax = plt.subplot(111)
    ax.set_xlim(0, 1)
    labels = []
    maxim = 0
    for i, arr in enumerate(values):
        # print(i)
        dim = np.arange(0, arr.shape[0], 1)
        label, = plt.plot(arr, linestyle='-', label=str(samples[i]), alpha=0.7)
        labels.append(label)
        plt.xticks(dim)
        maxim = max([max(arr), maxim])
    plt.ylim(ymin=0, ymax=(maxim * 1.1))
    plt.legend(handles=labels, loc=4)
    plt.grid()
    plt.show()
    plt.close()

class CompareDimReduction:

    components = 2

    DM_alpha = 1


    DM_t = 5


    ISOMAP_neigh = 10

    LLE_S_neigh = 10

    X = None
    classes = None
    title = None

    callback = None
    def __init__(self, X, classes, title, callback = None):
        self.X = X
        self.classes = classes
        self.title = title
        self.callback = callback

    def start(self):
        neigh_pre = []
        neigh_hit = []
        samples = []
        print('Started comparisson of:', self.title)
        print('\tEntries shape:', self.X.shape)
        print('\tClasses:', len(np.unique(self.classes)))


        # Diffusion Maps
        de = DiffusionMapEmbedding(alpha = self.DM_alpha, diffusion_time = self.DM_t, affinity='markov',
                                   n_components = self.components, metric='euclidean').fit_transform(self.X.copy())
        samples.append('Diffusion Maps')

        pts = plt.scatter(de[:, 0], de[:, 1], c = self.classes, linewidths=0)
        # print(classes)
        ax = plt.gca()
        fig = plt.gcf()
        fig.canvas.draw()
        selector = SelectFromCollection(ax, pts)
        def accept(event):
            nonlocal selector
            if event.key == "enter":
                print("Selected points:")
                if (self.callback != None):
                    self.callback(selector.ind)
                else:
                    print(selector.ind)
                selector.disconnect()
                # selector = SelectFromCollection(ax, pts)
                ax.set_title("Select")
                fig.canvas.draw()

        fig.canvas.mpl_connect("key_press_event", accept)
        plt.show()


        neigh_pre.append(nb.neighborhood_preservation(self.X, de))
        neigh_hit.append(nb.neighborhood_hit(de, self.classes))
        print('\tDiffusion Map finished')

        # Diffusion Maps2
        de = DiffusionMapEmbedding(alpha=self.DM_alpha, diffusion_time=self.DM_t, affinity='markov2',
                                   n_components=self.components, metric='euclidean').fit_transform(self.X.copy())
        samples.append('Diffusion Maps2')

        pts = plt.scatter(de[:, 0], de[:, 1], c=self.classes, linewidths=0)
        # print(classes)
        ax = plt.gca()
        fig = plt.gcf()
        fig.canvas.draw()
        selector = SelectFromCollection(ax, pts)

        def accept(event):
            nonlocal selector
            if event.key == "enter":
                print("Selected points:")
                if (self.callback != None):
                    self.callback(selector.ind)
                else:
                    print(selector.ind)
                selector.disconnect()
                # selector = SelectFromCollection(ax, pts)
                ax.set_title("Select")
                fig.canvas.draw()

        fig.canvas.mpl_connect("key_press_event", accept)
        plt.show()

        neigh_pre.append(nb.neighborhood_preservation(self.X, de))
        neigh_hit.append(nb.neighborhood_hit(de, self.classes))
        print('\tDiffusion Map2 finished')

        # ISOMAP
        iso = manifold.Isomap(self.ISOMAP_neigh, self.components).fit_transform(self.X)
        samples.append('ISOMAP')
        neigh_pre.append(nb.neighborhood_preservation(self.X, iso))
        neigh_hit.append(nb.neighborhood_hit(iso, self.classes))
        print('\tISOMAP finished')

        # LLE Standard
        lle_s = manifold.LocallyLinearEmbedding(self.LLE_S_neigh, self.components,
                                            eigen_solver='auto',
                                            method='standard').fit_transform(self.X)
        samples.append('LLE Standard')
        neigh_pre.append(nb.neighborhood_preservation(self.X, lle_s))
        neigh_hit.append(nb.neighborhood_hit(lle_s, self.classes))
        print('\tLLE Standard finished')

        print('Ploting')

        ## PLOT
        plot_preservation(neigh_pre, samples, self.title + " Neighborhood Preservation");
        plot_hit(neigh_hit, samples, self.title + " Neighborhood Hit");
