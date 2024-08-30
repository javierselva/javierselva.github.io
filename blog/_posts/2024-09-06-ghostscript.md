---
layout: post
section-type: post
has-comments: False
title: Using ghostscript to hide private data in PDFs
category: tnt
tags: ["tips","tricks","pdf","ghostscript","privacy"]
---

This something I've frankly had to do several times. Some entity requires a sepecific information from you, information you've got in a PDF. However, that file does contain some other sensitive information (e.g., bank transactions) you do not wish to share. It would be very helpful, then, to be able to extract that one or two pages you want to share.

First step is [installing ghostscript](https://ghostscript.readthedocs.io/en/latest/Install.html). Once that's done, you can simply use the following line:

```bash
gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=22 -dLastPage=36 -sOutputFile=outfile.pdf inputfile.pdf 

```

This is quite straigtforward, you can easily find that in [stackoverflow](https://stackoverflow.com/questions/10228592/splitting-a-pdf-with-ghostscript). However, I've sometimes wanted to **hide private information** that was contained in the same page as the data I needed to share. For this, I always thought the best solution was to hide it as they do with top secret documents in the movies: by simply **crossing it out with black ink**. For this particular task I find [Xournal](https://xournalpp.github.io/) to be a very nice tool. Not only does it allow you to annotate a PDF, I have also used it to fill in documents which are not forms or adding images. In this case, you can use it to cross out the information you want hidden. There is one catch though, apparently what Xournal does is adding one layer on top of the actual PDF. What this means in practice is that someone could simply **select all the text from the document and copy it**, even if it is behind a black blob. But do not despair, ghostcript is here to help. 


Simply add the flag `-dNoOutputFonts` to the line above, which will generate an image-like PDF where slecting and copying text will be no longer possible.

Now you can easily share the relevant information without disclosing more than you need to. 