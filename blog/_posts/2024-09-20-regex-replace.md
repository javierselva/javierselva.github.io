---
layout: post
section-type: post
has-comments: False
title: Using regex to replace multiple instances.
category: tnt
tags: ["regex","replace","tips","tricks"]
---

This may have happened to me more than one time. I was editing a Latex file and wanted, for instance, to remove a bunch of `\textbf{bold}` text but keep it as plain text, or accept a bunch of comments (I generally mark working text in red to be later edited/removed/accepted, such as `\textcolor{red}{lorem ipsum...}`). My first idea was to use some regex expression to find all these, and somehow parametrize the content so it could be referenced from the replace field of the text editor. In this way, every find-replace would be unique depending on the content each instantiation of the regex in the find had found.

To explain the solution I'm going to be giving the following example. I have a latex table with a bunch of numbers, some in bold, highlighting the best results for a column. After a couple new experiments, I need to change which elements of the table are highlighted in bold, so I need to revert it all to plain text so I can start again.

![Screenshot of a text editor showing a bunch of decimal numbers separated by the ampersand symbol. Some numbers are in bold latex format, for instance \textbf{3.258}](/assets/img/tnt/latex_regex.png)

Very easily you can find all those instances by using:

```
\\textbf{[\d].[\d]*}
```

(The starting double \\ is to scape the \ from \textbf{}). Now, somehow, I need to be able to tell the replace field to replace each instance with the matched contents in `[\d].[\d]*`. A quick google search finds [this post](https://www.overleaf.com/learn/how-to/Can_I_use_regular_expressions_for_%22replace_with%22%3F) from Overleaf where they explain it should be as easy as doing:

```
Replace: /([a-z])-([a-z])/
With: /$1$2/
```

And it really is that easy. But for the life of me, I didn't know what was I doing wrong, but it was not working. Turns out, I thought the parenthesis were part of a specific example, as that hyphen between both `[a-z]`, but turns out the **[parenthesis are fundamental to capture a regex group](https://www.regular-expressions.info/refcapture.html)**, which is what we're doing here.

So coming back to our example, the solution is:

```
Find: \\textbf{([\d]).([\d]*)}
Replace: $1.$2 
```
![Screenshot of a text editor showing a bunch of decimal numbers separated by the ampersand symbol. Now all numbers appear in plain text](/assets/img/tnt/latex_regex2.png)