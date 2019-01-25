Course project for ["Analysing Software using Deep Learning"](https://www.sola.tu-darmstadt.de/index.php?id=13101). See the [project description](https://www.sola.tu-darmstadt.de/fileadmin/user_upload/Group_SOLA/Teaching/summer_2017/ASDL/project_description_20170529.pdf) for details. The code is based on Michael Pradels [repository](https://github.com/michaelpradel/ASDL2017).

The task was to fill missing code fragments using deep learning. This could for example be used in code editors to make programming faster. In the following example the network would probably predict _i++_ for the missing part.
```
for (i = 0; i < cars.length; ...) { 
  text += cars[i] + "<br>";
}
```

To aqcuire this goal 1000 code files were provided that could be used for training. 

The best accuracy of 0.645 on the test set was reached with a recurrent neural network and by using both the prefix as well as the suffix of the missing code fragment.




