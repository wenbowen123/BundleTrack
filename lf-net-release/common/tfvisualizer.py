# -*- coding: utf-8 -*-
import sys
from IPython.display import clear_output, Image, display, HTML
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import pandas as pd
from io import BytesIO
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>"%size, 'utf-8')
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def facets_confusion_matrix(images, actuals, predictions, disp_height=800):
    N,C,H,W = images.shape
    assert(N == len(actuals))
    assert(N == len(predictions))
    atlas_img = tile_images(images, padsize=0)
    imsave('atlas.jpg', atlas_img)
    df = pd.DataFrame({"prediction": predictions, "actual": actuals})
    jsonstr = df.to_json(orient='records')
    
    HTML_TEMPLATE = """
        <head>
        <link rel="import" href="/nbextensions/facets-dist/facets-jupyter.html"></link>
        </head>
        <facets-dive id="elem" height="{disp_height}" sprite-image-width="{img_width}" sprite-image-height="{img_height}" atlas-url="atlas.jpg"></facets-dive>
        <script>
          var data = JSON.parse('{jsonstr}');
          document.querySelector("#elem").data = data;
        </script>"""
    html = HTML_TEMPLATE.format(jsonstr=jsonstr, disp_height=disp_height, img_width=W, img_height=H)
    display(HTML(html))

def convert_tile_image(inputs, padsize=1):
    # inputs.shape = [B,H,W,C]
    with tf.name_scope('TILE_IMAGE'):
        B = tf.shape(inputs)[0]
        n = tf.cast(tf.ceil(tf.sqrt(tf.cast(B, tf.float32))), tf.int32)
        padding = [[0,n**2-B],[0,padsize],[0,padsize],[0,0]]
        outputs = tf.pad(inputs, padding)
        oshp = tf.shape(outputs)
        H = oshp[1]
        W = oshp[2]
        C = oshp[3]
        outputs = tf.reshape(outputs, (n, n)+(H, W, C))
        outputs = tf.transpose(outputs, (0,2,1,3,4))
        outputs = tf.reshape(outputs, (n*H, n*W, C))
        # outputs = tf.squeeze(outputs)
        return outputs[None] # to save with tf.summary.image, tensor have to keep 4D

class TBLogger(object):
    """Logging in tensorboard without tensorflow ops."""
    """copy from <https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514>"""

    def __init__(self, _sentinel=None, log_dir=None, writer=None):
        """Creates a summary writer logging to log_dir."""
        if writer is not None:
            self.writer = writer
        elif log_dir is not None:
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            raise Exception('Empty args (log_dir or writer)')

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_images(self, tag, images, step, order_bgr=False, order_nchw=False):
        """Logs a list of images."""
        assert images.ndim == 4
        im_summaries = []

        for nr, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            # img shape = [H,W,C]

            if order_nchw:
                img = img.transpose(1,2,0) # convert HWC order
            if img.shape[2] == 3 and order_bgr:
                img[...,[0,1,2]] = img[...,[2,1,0]]
            if img.shape[2] == 1:
                img = img[...,0] # convert [H,W,1] to [H,W]

            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()
        
    def log_image(self, tag, image, step, order_bgr=False):
        """Logs a list of images."""
        assert image.ndim <= 3
        im_summaries = []
        img = image.copy()

        # Write the image to a string
        s = BytesIO()

        if img.ndim == 2:
            pass
        elif img.ndim == 3:
            if order_bgr and img.shape[2] == 3:
                img[...,[0,1,2]] = img[...,[2,1,0]]
            elif img.shape[2] == 1:
                img  = img[...,0] # convert [H,W,1] to [H,W]

        plt.imsave(s, img, format='png')

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])
        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag='%s' % (tag),
                                             image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_figure(self, tag, fig, step):
        # Write the image to a string
        # fig = plt.gcf()
        
        im_summaries = []
        s = BytesIO()
        plt.figure(fig.number)
        plt.savefig(s, format='png')

        w = int(fig.get_figwidth() * fig.dpi)
        h = int(fig.get_figheight() * fig.dpi)
        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=h,
                                   width=w)
        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag='%s' % (tag),
                                             image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

def log_scalar(writer, tag, value, step):
    TBLogger(writer=writer).log_scalar(tag, value, step)

def log_images(writer, tag, images, step, order_bgr=False, order_nchw=False):
    TBLogger(writer=writer).log_images(tag, images, step, order_bgr=order_bgr, order_nchw=order_nchw)

def log_image(writer, tag, image, step, order_bgr=False):
    TBLogger(writer=writer).log_image(tag, image, step, order_bgr=order_bgr)

def log_figure(writer, tag, fig, step):
    TBLogger(writer=writer).log_figure(tag, fig, step)

def log_histogram(writer, tag, values, step, bins=1000):
    TBLogger(writer=writer).log_histogram(tag, values, step, bins=bins)
