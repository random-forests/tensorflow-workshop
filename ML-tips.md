# Approach problems in this order

1. Carefully think about what you want to predict, what success means, and how you will **evaluate it**.

2. Carefully collect & clean data, and write code to detect errors in your data processing pipeline.

3. Begin with simple features, and a baseline model (say, a linear classifier).

4. Run end-to-end experiments. Are your predictions useful? Do they make sense? 

5. **Analyze errors. What kind of mistakes does your model make? Why?** What can you do to prevent them? Get to know your data. 

6. Improve your features. If you use complex features, are they **likely to generalize** (e.g., also be useful on your test data)?

7. Consider using a more complex model. Switch from a linear classifier to a DNN (**be careful not to overfit**). If you want to simulate production code, you get to use your test data only once. 

Tip: don't agonize over the number of layers, or neurons per layer (it's fun to hack on these, but it's usually not that important). There are **many possible architectures** that will give similar accuracy.

Remember: the goal is to develop a model that works well in production. 

### Design for **simplicity and reliable** first, and you're more likely to get a good result.

# What kind of accuracy improvement can I expect from...?

Tip: focus on your data, and your features. They're almost always more important than tuning your model. Here's the relative accuracy improvment you can expect from each activity.

### Improving your **features**
* Reasonable features +50%
* Tuned features +20%
* Amazing features +10%

### Improving your **model**
* Baesline model +50%
* Tuned model +10%
* Amazing model +5% 
