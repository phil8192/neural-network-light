# load saved data 
errors <- read.table("/tmp/errors.csv", header=F, sep=",", 
    col.names=c("training", "testing"))

# plot training vs testing mse
par(mfrow=c(2, 1), oma=c(0, 0, 0, 0), omi=c(0, 0.4, 0, 0), mar=c(0, 0, 0, 0))
plot(errors$training, type="l")
plot(errors$testing, type="l")

