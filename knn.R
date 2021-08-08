library(ggplot2)

#Files used to get the texts
testfile = "../test.txt"
trainfile = "../train.txt"

#This function expects an argument to be relative path to the pixel data. The output is a dataframe
read_digits = function(filepath){
  df = read.table(filepath)
  colnames(df) = c("class_label_digit",paste0("pixel", seq(2:ncol(df)))) #paste0 faster than paste
  return (df)
}

#Reading both the dataset, training and test.
trainfile = read_digits(trainfile)
testfile = read_digits(testfile)


#This function expects a pixelated dataframe as argument. Output is suppose to be the images of multiple pixelated digits 0-9
display_images= function(x){
  x = split(x, x[,1])
  par(mar = c(1,1,1,1))
  par(mfrow=c(3,4),xaxt="n", yaxt = "n")
  for (i in 1:length(x)){
    xs = as.data.frame(x[i])
    xs = sapply(xs[,-1], mean)
    xs = as.matrix(xs)
    dim(xs) = c(16,16)
    image(xs,col=grey(seq(1,0,length=256)),ylim =c(1,0))
    title(paste0('Number ',i-1 ), font.main=1)
  }
}
pixel_images = display_images(trainfile)
sd_pixels = apply(trainfile[,-1], 2, var)
min(sd_pixels)
max(sd_pixels)
which.min(sd_pixels)
which.max(sd_pixels)

#This function expect a vector of digits. Its main goal is to randomize ties or pick the highest frequency digits.
#Output is a single digit from 0-9

is_tie = function(x){ # pass in the vector of digits
  tbl = table(x)
  df = as.data.frame(tbl)
  df[,1] = as.numeric(as.character(df[,1]))
  if (length(unique(df[,2]))>1){
    max_x = max(df[,2])
    ind = which(df[,2]==max_x)#check which numbers is the count
    digit = df[,1][ind]
    if (length(digit)>1){
      digit = sample(digit,1)
    }
  }
  else{
    digit = x[1] #if they are all the same return
  }
  return (digit)
}

#This function expects a distance matrix as its first argument and k and the training digits from the dataframe
#Its job is to look predict which number is the correct number. Used "is_tie" function to isolate the final digit.
#Output is suppose to be a vector of digits

predict_knn = function(dist_metric, k, digits){
  store_digits = vector()
    for (i in 1:ncol(dist_metric)){ #Look for the numbers closet to the prediction
    label = order(dist_metric[,i]) #get the smallest distance by columns so we don't get train
    k_label = label[1:k]#get the smallest distance by index up to k
    k_label = digits[k_label]
    estim_digit = is_tie(k_label)
    store_digits = append(store_digits,estim_digit)
  }
  return (store_digits)
}

#This function inputs are a dataframe, the number of nearest neigbors the user wants to calculate, and a mathematical
#method to calculate distance: euclidean, manhattan, maximum, etc. This function calculates the error rate of the folds
#in k-nearest neigbors. Output is a vector of the 10 fold error rates

cv_error_rate = function(train_set,k, dist_metric){
  ind = sample(nrow(train_set), nrow(train_set)) #randomize index
  full_train_set = train_set[ind,]
  train_set = full_train_set[,-1]#only need for the distance
  label = full_train_set[,1]
  thedist = as.matrix(dist(train_set, method = dist_metric))
  folds = split(ind, rep(1:floor(nrow(train_set)/10),
                                      each = floor(nrow(train_set)/10),
                                      length.out = nrow(train_set))) #split by index
  folds = folds[-11] #the extra list of a single row. Assumed that only this training set is given
  error = sapply(folds,function(fold){
    #applying each fold to the 9 fold
    tdist = thedist[-fold,fold] #fold are the index, -fold means all the other folds
    train_label = label[-fold]
    true_label = label[fold]
    error_rate = sum(true_label != predict_knn(tdist, k, train_label))/length(true_label)
  }
  )
  avg_error = sum(error/10)
  return (avg_error)
}

#Using training set for all the distance methods with k = 15
#Adjusted the "cv_error_knn" function in order to run 1 to 15 more effieciently. Instead of calculating the distance
#in this function I did in the begining of everything to make sure it runs less than 5minutes. It takes approximately
#1 minute and 30 seconds to run each method.

adjusted_cv_error = function(thedist,folds,label, k){
  error = sapply(folds,function(fold){
    #applying each fold to the 9 fold
    tdist = thedist[-fold,fold] #fold are the index, -fold means all the other folds
    train_label = label[-fold]
    true_label = label[fold]
    error_rate = sum(true_label != predict_knn(tdist, k, train_label))/length(true_label)
  }
  )
  avg_error = sum(error)/10
  return (avg_error)
}

k_multiple_errors = function(train_set, k, dist_metric){
  error_rates = vector("list", length = 10)
  ind = sample(nrow(train_set), nrow(train_set))
  full_train_set = train_set[ind,]
  train_set = full_train_set[,-1]
  label = full_train_set[,1]
  thedist = as.matrix(dist(train_set, method = dist_metric))
  folds = split(ind, rep(1:floor(nrow(train_set)/10),
                         each = floor(nrow(train_set)/10),
                         length.out = nrow(train_set)))
  folds = folds[-11]
  for (i in 1:k){
    error_rates[[i]] = adjusted_cv_error(thedist,folds,label,i)
  }

  return(error_rates)
}

euc = unlist(k_multiple_errors(trainfile,15,"euclidean"))
man = unlist(k_multiple_errors(trainfile,15,"manhattan"))
kowski = unlist(k_multiple_errors(trainfile,15,"minkowski"))
maxim = unlist(k_multiple_errors(trainfile, 15, "maximum"))
berra = unlist(k_multiple_errors(trainfile, 15,"canberra"))

#To pick the best k, look at k that has decent variation and low error rate. High variation is not always bad since
#it leads to lower bias.

all_errors = as.data.frame(cbind(euc,man,kowski,berra,maxim))

error_5 = ggplot(all_errors, aes(x=1:15)) + geom_point(aes(y = euc), size = .8) +
  geom_line(alpha = .5, aes(y=euc, color = "Euclidean")) +
  geom_point(aes(y= kowski), size = .8) +
  geom_line(linetype= "dashed", aes(y = kowski, colour = "Minkowski")) +
  scale_x_continuous(breaks = c(seq(1,15,2))) +
  labs(title = "Cross Validation Error Rates ", x = "k-Nearest Neigbors", y = "Error Rates", subtitle = "10 Fold Training Set") +
  guides(color = guide_legend("Distance Methods")) +
  scale_color_manual(values = c("#D55E00","#0072B2")) +
  theme(plot.title = element_text(hjust = 0.5, size = 10, face = "bold.italic")) +
  theme(plot.subtitle = element_text(hjust = 0.5, size = 8, face = "italic"))
ggsave("error_5.png")
#Best k= 9, either euclidean or minkowski methods.


#Since the previous predict_knn is meant to be applied to the cross validation,
#I created a different predict_knn to do number 6. This is function is very similar to predict_knn and cross_validation
#since I combined both ideas into a single functipn
#Since we have the best models, minkowski and euclidean. We will only use these two methods for both files

adjust_predict_knn = function(train, test, dist_metric, k){
  store_digits = vector()
  error = vector("list", length = k)
  df = rbind(test,train)
  test_labels = test[,1]
  train_labels = train[,1]
  thedist = as.matrix(dist(df[,-1], method = dist_metric))
  thedist = thedist[1:nrow(test), (nrow(test)+1):nrow(df)]

  for (j in 1:k){
    for (i in 1:nrow(test)){
      label = order(thedist[i,])
      k_label = label[1:j]
      k_label = train_labels[k_label]
      estim_digit = is_tie(k_label)
      store_digits = append(store_digits,estim_digit)
    }
    error[[j]] = sum(test_labels != store_digits)/length(test_labels)
    store_digits = vector()
  }
  return (error)
}

euc_6 = unlist(adjust_predict_knn(trainfile,testfile, "euclidean", 15))
kowski_6 = unlist(adjust_predict_knn(trainfile,testfile, "minkowski", 15))
error_6 = as.data.frame(cbind(euc_6,kowski_6))

#Graph for number 6
error_6g = ggplot(error_6, aes(x=1:15)) + geom_point(aes(y = euc_6), size = .8) +
  geom_line(alpha = .5, aes(y=euc_6, color = "Euclidean")) +
  geom_point(aes(y= kowski_6), size = .8) +
  geom_line(linetype= "dashed", aes(y = kowski_6, colour = "Minkowski")) +
  scale_x_continuous(breaks = c(seq(1,15,2))) +
  labs(title = "K-Nearest Neighbors", x = "Range of K", y = "Error Rates",subtitle = "Training and Test Set") +
  guides(color = guide_legend("Distance Methods")) +
  scale_color_manual(values = c("#D55E00","#0072B2")) +
  theme(plot.title = element_text(hjust = 0.5, size = 10, face = "bold.italic")) +
  theme(plot.subtitle = element_text(hjust = 0.5, size = 8, face = "italic"))
ggsave("error_6g.png")


#Similar to "cv_error_rate" function, instead of returning the error rate, this function returns a list of
#numbers that the prediction got wrong
best_wrong = function(train_set,k, dist_metric){
  wrong_nums = vector()
  ind = sample(nrow(train_set), nrow(train_set)) #randomize index
  full_train_set = train_set[ind,]
  train_set = full_train_set[,-1]#only need for the distance
  label = full_train_set[,1]
  thedist = as.matrix(dist(train_set, method = dist_metric))
  folds = split(ind, rep(1:floor(nrow(train_set)/10),
                         each = floor(nrow(train_set)/10),
                         length.out = nrow(train_set))) #split by index
  folds = folds[-11] #the extra list of a single row. Assumed that only this training set is given
  error = sapply(folds,function(fold){
    #applying each fold to the 9 fold
    tdist = thedist[-fold,fold] #fold are the index, -fold means all the other folds
    train_label = label[-fold]
    true_label = label[fold]
    numbers = which(true_label!= predict_knn(tdist, k, train_label))
    wrong_nums = append(wrong_nums,true_label[numbers])
  }
  )
  return (error)
}

#Checking which number the algorithm is most likely to get wrong
bestmodel_wrongdigits = best_wrong(trainfile,k=9,"minkowski")
bestmodel_wrongdigits = unlist(bestmodel_wrongdigits)
best_model = as.data.frame(bestmodel_wrongdigits)
ggplot(best_model, aes(x = bestmodel_wrongdigits, col = bestmodel_wrongdigits)) + geom_bar(position = "identity") +
  scale_x_continuous(breaks = c(0:9)) +
  labs(x= "Digits 0-9", y = "Frequency of Miscount",
       title = "Failed Prediction from the Best Model", subtitle = "Minkowski Method, K = 9") +
  theme(plot.title = element_text(hjust = 0.5, size = 10, face = "bold.italic")) +
  theme(plot.subtitle = element_text(hjust = 0.5, size = 8, face = "italic"))
