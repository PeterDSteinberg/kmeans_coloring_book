<html>
  <head>
    <meta charset='utf-8'>
    <meta http-equiv="X-UA-Compatible" content="chrome=1">
    <meta name="description" content="Kmeans coloring book : Takes any image through the kmeans algorithm to make a white and black line simplified picture, like a coloring book.">
    <link rel="stylesheet" type="text/css" media="screen" href="stylesheets/stylesheet.css">
    <title>Kmeans coloring book</title>
  </head>
  <body>
    <div id="header_wrap" class="outer">
        <header class="inner">
          <a id="forkme_banner" href="https://github.com/PeterDSteinberg/kmeans_coloring_book">View on GitHub</a>
          <h1 id="project_title">Kmeans coloring book</h1>
          <h2 id="project_tagline">Coloring book type of drawing from pictures.</h2>
            <section id="downloads">
              <a class="zip_download_link" href="https://github.com/PeterDSteinberg/kmeans_coloring_book/zipball/master">Download this project as a .zip file</a><br>
              <a class="tar_download_link" href="https://github.com/PeterDSteinberg/kmeans_coloring_book/tarball/master">Download this project as a tar.gz file</a>
            </section>
        </header>
    </div>
    <div id="main_content_wrap" class="outer">
      <section id="main_content" class="inner">
        <p>The screenshots below show 2 sets of input images and output sketchs from kmeans.  The hatched triangle is a user selected training region to reduce the effect of extraneous data in images (excess background, washed out areas because of lighting, etc....).</p>
        <p><img src="http://hiswmm1.s3.amazonaws.com/Airplane_kmeans_coloring.png" alt=""></p>
        <div>Original airplane photo credit: 7-themes.com</div>
        <p><img src="http://hiswmm1.s3.amazonaws.com/Kmeans_coloring_book_example.png" alt=""></p>
        <div style="font-size:8px;">Curious George image credit: http://www.pbs.org/about/news/archive/2013/curious-george-spring/</div>
<h3>
<a id="kmeans-to-make-coloring-book-pages" class="anchor" href="#kmeans-to-make-coloring-book-pages" aria-hidden="true"><span class="octicon octicon-link"></span></a>Kmeans to Make Coloring Book Pages</h3>
<pre><code>
$ git clone https://github.com/PeterDSteinberg/kmeans_coloring_book
$ cd kmeans_coloring_book
$ # move your input photos to ./raw_images relative to pwd at this point
$ python -i kmeans_coloring_book.py # starts the interactive plotter / classifier shown below:
</code></pre>
<p><img src="http://hiswmm1.s3.amazonaws.com/kmeans_cli_picture.png" alt=""></p>
<p>Make a coloring book out of any image(s) you have in ./raw_images directory.</p>

<p>Adjust the number of colors (k in kmeans).</p>

<p>I found this script useful qualitatively for exploring the behavior of k-means when the training data (image color vectors) have unequal numbers of observations of each class.</p>  
<p>It is interesting to watch the effect of overfitting (making a coloring book picture that is too intricate)</p>
<h3><a id="support-or-contact" class="anchor" href="#support-or-contact" aria-hidden="true"><span class="octicon octicon-link"></span></a>Support or Contact</h3><p>peterdsteinberg [at]    g[no space]mail [dot] com</p></section></div>
