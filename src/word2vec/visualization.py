import matplotlib.pyplot as plt

from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.preprocessing import scale

import config

def draw_vectors(x, y, word_texts, vis_fn):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(x, y)
    
    for i, txt in enumerate(word_texts):
        ax.annotate(txt, (x[i], y[i]))

    fig.savefig(vis_fn)

def get_tsne_projection(word_vectors):
    tsne = TSNE(n_components=2)
    return scale(tsne.fit_transform(word_vectors))


def visualize_embeddings(embeddings, index2word, word_count, vis_fn=config.PROJECTION_FILE):
    word_vectors = embeddings[1: word_count + 1]
    words        = index2word[1: word_count + 1]

    word_tsne    = get_tsne_projection(word_vectors)
    draw_vectors(word_tsne[:, 0], word_tsne[:, 1], words, vis_fn)
