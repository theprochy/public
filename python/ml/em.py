import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
plt.ion()

# config
plot = True

# load images
shape = False
imgs = np.zeros((50, 289, 250, 3))
shapes = np.zeros((50, 289, 250))
i = 0
for im_path in glob.glob("data/*.png"):
    im = plt.imread(im_path)
    if "seg" in im_path:
        shapes[i] = im
        i = i + 1
    elif "hand" in im_path:
        imgs[i] = im

init0 = np.zeros((289, 250), dtype=bool)
init0[:, 224:249] = True
A = np.arange(-124.5, 125.5) ** 2
center = np.sqrt(A[:, None] + A) < 63
init1 = np.zeros_like(shapes[0], dtype=bool)
init1[39:289, :] = center

# EM algorithm
print("Starting EM algorithm")
alpha = np.zeros((50, 289, 250))
p_1_u = im[:, :, 0]
mu_0 = np.average(imgs[:, init0], axis=1)
mu_1 = np.average(imgs[:, init1], axis=1)
sigma_0 = np.zeros((50, 3, 3))
sigma_1 = np.zeros((50, 3, 3))
for i in range(50):
    sigma_0[i] = np.cov(imgs[0, init0, :].T)
    sigma_1[i] = np.cov(imgs[0, init1, :].T)
stop, i = False, 0
while True:
    print("Iteration %d" % (i +1))
    i = i + 1
    # E-step
    for j in range(50):
        p_x_0 = (multivariate_normal.pdf(imgs[j], mean=mu_0[j], cov=sigma_0[j]))
        p_x_1 = (multivariate_normal.pdf(imgs[j], mean=mu_1[j], cov=sigma_1[j]))
        alpha[j] = (p_x_1 * p_1_u) / (p_x_0 * (1 - p_1_u) + p_x_1 * p_1_u)

    # M-step
    p_1_u_old = p_1_u
    p_1_u = np.sum(alpha, axis=0) / 50
    if np.sum(np.abs(p_1_u - p_1_u_old)) / 50 < 0.05:
        stop = True
    for j in range(50):
        X = imgs[j].reshape(-1, 3)
        mu_0[j] = np.average(X, weights=(1 - alpha[j]).reshape(-1, 1)[:, 0], axis=0)
        mu_1[j] = np.average(X, weights=alpha[j].reshape(-1, 1)[:, 0], axis=0)
        sigma_0[j] = np.cov(X.T, aweights=(1 - alpha[j]).reshape(-1, 1).T[0])
        sigma_1[j] = np.cov(X.T, aweights=alpha[j].reshape(-1, 1).T[0])
    if stop:
        break
print("EM stopped")
plt.figure()
plt.title("Final value of u")
plt.imshow(p_1_u, cmap=plt.get_cmap("gray"))
plt.show()
em_correct = 0
for i in range(len(imgs)):
    em_correct = em_correct + np.sum((alpha[i] > 0.5) == shapes[i])
print("EM accuracy %.2f" % (em_correct * 100 / (50 * 289 * 250)))

# baseline methods
print("Running baseline methods")
if plot:
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
k_means_correct, gm_correct = 0, 0
for i in range(len(imgs)):
    print("Processing image %d" % i)
    img = imgs[i]
    X = img.reshape((-1, 3))
    means_init = np.zeros((2, 3))
    means_init[0] = np.average(img[init0], axis=0)
    means_init[1] = np.average(img[init1], axis=0)
    kmeans = KMeans(n_clusters=2, n_init=1, random_state=0, init=means_init).fit(X)
    kmeans_y = kmeans.labels_.reshape((289, 250))
    k_means_correct = k_means_correct + np.sum(kmeans_y == shapes[i])
    gm = GaussianMixture(n_components=2, random_state=0, means_init=means_init).fit(X)
    gm_y = gm.predict(X).reshape((289, 250))
    gm_correct = gm_correct + np.sum(gm_y == shapes[i])
    if plot:
        plt.tight_layout()
        plt.suptitle("img %d" % i)
        ax1.title.set_text("kmeans")
        ax2.title.set_text("mixture")
        ax3.title.set_text("EM")
        ax4.title.set_text("orig")
        ax1.imshow(kmeans_y, cmap=plt.get_cmap("gray"))
        ax2.imshow(gm_y, cmap=plt.get_cmap("gray"))
        ax3.imshow((alpha[i] > 0.5).astype(int), cmap=plt.get_cmap("gray"))
        ax4.imshow(imgs[i])
        plt.show()
        plt.pause(1)
print("k-means accuracy: %.2f" % (k_means_correct * 100 / (50 * 289 * 250)))
print("gm accuracy: %.2f" % (gm_correct * 100 / (50 * 289 * 250)))
