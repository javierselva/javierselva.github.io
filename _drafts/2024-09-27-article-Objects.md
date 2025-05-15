Object detection

This post has an image at the beginning that I think provides an overview of recent methods:
https://www.v7labs.com/blog/yolo-object-detection

Generally, single-shot object detection is better suited for real-time applications, while two-shot object detection is better for applications where accuracy is more important.

Intersection over Union.
The minimun union is the ground truth area. From there, the union grows with the detection area beyond that ground truth. So in this sense, the larger the union (at least beyond that minimum), the worse the metric should be. Regarding the intersection, again, the best possible outcome is that the intersection is the ground truth area. The larger it is the better, up to that value. As the detection gets worse (meaning, further away from the actual ground truth), the metric gets worse. So it is very intuitive, the further away from gt (skewed in space or lager) the detection gets, this will increase the union, without increasing (or even decreasing) the intersection, hence reducing the value of the ratio.

Let's think of some examples:
	- The detection is smaller and completely inside the gt. Union is the gt area, while intersection is small. As the detection grows, but still within the gt, the intersection gets larger. IoU progressively grows (numerator growing, denominator fixed), up to a maximum of 1 when intersection area matches union area.
	- Is it possible to get 1 other than in the fully matched case? I don't think so, if we start there, and either a) make the detection smaller, b) make the detection bigger or c) move it arround, we change at least one of those properties, making a) the intersection smaller, hence IoU <1; b) the union larger, hence IoU also < 1; or c) the union larger and the intersection smaller, hence IoU < 1 even more.
	- It cannot be larger than 1 because that would entail that there is more area in common than with both combined. https://en.wikipedia.org/wiki/Jaccard_index according to wikipedia, this range makes sense.


Other more complex metrics, but I hear they're used a lot, are Precision-Recall curves, but I am not sure I fully understand them
https://stats.stackexchange.com/questions/601131/average-precision-ap-for-object-detection-huge-confusion
