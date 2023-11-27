pathin=c("~/download_sc/Pancreas/plot/muraro_other/muraro_other.csv")
pathout<-c("~/download_sc/Pancreas/plot/muraro_other/muraro_other.pdf")


acc<-read.csv("~/download_sc/Pancreas/plot/muraro_other/muraro_other.csv")


colnames(acc)[1]<-c('canshu')

acc<-as.data.frame(acc)
library(ggplot2)
library(reshape2)
library(ggsignif)
library("ggpubr")

# 把数据转换成ggplot常用的类型（长数据）

df<-acc
df = melt(df)


cols<-c('#E64E00','#65B48E','#E6EB00','#E64E00')
pal<-colorRampPalette(cols)
#使用ggplot2包生成箱线图
P1 <- ggplot(df,aes(x=variable,y=value,fill=variable))+ #'fill='设置填充颜色
  stat_boxplot(geom = "errorbar",width=0.15,aes(color="black"))+ #由于自带的箱形图没有胡须末端没有短横线，使用误差条的方式补上
  geom_boxplot(alpha = 1,size=0.5,fill="white",outlier.color = "black",outlier.fill="white")+ #size设置箱线图的边框线和胡须的线宽度，fill设置填充颜色，outlier.fill和outlier.color设置异常点的属性
  #geom_boxplot(size=0.5,fill="white",outlier.fill="white",outlier.color="white")+ #size设置箱线图的边框线和胡须的线宽度，fill设置填充颜色，outlier.fill和outlier.color设置异常点的属性
  #geom_jitter(aes(fill=variable),width =0.2,shape = 21,size=2.5)+ #设置为向水平方向抖动的散点图，width指定了向水平方向抖动，不改变纵轴的值
  geom_point(aes(colour = factor(canshu),shape=canshu))+
  #scale_fill_manual(values = pal(12))+  #设置填充的颜色
  #scale_color_manual(values=c("black","black","black"))+ #设置散点图的圆圈的颜色为黑色
  ggtitle("ACCURACY")+ #设置总的标题
  theme_bw()+ #背景变为白色
  theme(
    #legend.position="none", #不需要图例
    axis.text.x=element_text(colour="black",family="Times",size=14), #设置x轴刻度标签的字体属性
    axis.text.y=element_text(family="Times",size=14,face="plain"), #设置x轴刻度标签的字体属性
    axis.title.y=element_text(family="Times",size = 14,face="plain"), #设置y轴的标题的字体属性
    axis.title.x=element_text(family="Times",size = 14,face="plain"), #设置x轴的标题的字体属性
    plot.title = element_text(family="Times",size=15,face="bold",hjust = 0.5), #设置总标题的字体属性
    panel.grid.major = element_blank(), #不显示网格线
    panel.grid.minor = element_blank())+
  ylab("ACCURACY")+xlab("METHODS") +#设置x轴和y轴的标题)
  theme(axis.text.x = element_text(angle = 30,vjust = 0.5) )      # x轴刻度改为倾斜90度，防止名称重叠



P1

###箱型图加点图
ggplot(df,aes(x=variable,y=value,fill=variable))+
  geom_boxplot(alpha = 1,              # 透明度
               outlier.color = "black" # 外点颜色
  )+
  geom_point(aes(colour = factor(canshu),shape=canshu,size=1.5))+
  ggtitle("ACCURACY")+ #设置总的标题
  theme_bw()+ #背景变为白色
  theme(
    #legend.position="none", #不需要图例
    axis.text.x=element_text(colour="black",family="Times",size=14), #设置x轴刻度标签的字体属性
    axis.text.y=element_text(family="Times",size=14,face="plain"), #设置x轴刻度标签的字体属性
    axis.title.y=element_text(family="Times",size = 14,face="plain"), #设置y轴的标题的字体属性
    axis.title.x=element_text(family="Times",size = 14,face="plain"), #设置x轴的标题的字体属性
    plot.title = element_text(family="Times",size=15,face="bold",hjust = 0.5), #设置总标题的字体属性
    panel.grid.major = element_blank(), #不显示网格线
    panel.grid.minor = element_blank())+
  ylab("ACCURACY")+xlab("METHODS") +#设置x轴和y轴的标题)
  theme(axis.text.x = element_text(angle = 30,vjust = 0.5) )      # x轴刻度改为倾斜90度，防止名称重叠

##不同数据集不同形状
ggplot(df,aes(x=variable,y=value,fill=variable))+
  geom_boxplot(alpha = 1, outlier.color = "black" )+
  geom_point(aes(colour = factor(canshu),shape=canshu))+
  scale_shape_manual(values = c(1:12))+
  ggtitle("ACCURACY")+ #设置总的标题
  theme_bw()+ #背景变为白色
  theme(
    #legend.position="none", #不需要图例
    axis.text.x=element_text(colour="black",family="Times",size=14), #设置x轴刻度标签的字体属性
    axis.text.y=element_text(family="Times",size=14,face="plain"), #设置x轴刻度标签的字体属性
    axis.title.y=element_text(family="Times",size = 14,face="plain"), #设置y轴的标题的字体属性
    axis.title.x=element_text(family="Times",size = 14,face="plain"), #设置x轴的标题的字体属性
    plot.title = element_text(family="Times",size=15,face="bold",hjust = 0.5), #设置总标题的字体属性
    panel.grid.major = element_blank(), #不显示网格线
    panel.grid.minor = element_blank())+
  ylab("ACCURACY")+xlab("METHODS") +#设置x轴和y轴的标题)
  theme(axis.text.x = element_text(angle = 30,vjust = 0.5) )      # x轴刻度改为倾斜90度，防止名称重叠

#################3

P1 <- ggplot(df,aes(x=variable,y=value,fill=canshu))+ #'fill='设置填充颜色
  stat_boxplot(geom = "errorbar",width=0.15)+ #由于自带的箱形图没有胡须末端没有短横线，使用误差条的方式补上
  geom_boxplot(alpha = 0.6,size=1,fill="white",outlier.color = "black",outlier.fill="white")+ #size设置箱线图的边框线和胡须的线宽度，fill设置填充颜色，outlier.fill和outlier.color设置异常点的属性
  #geom_boxplot(size=0.5,fill="white",outlier.fill="white",outlier.color="white")+ #size设置箱线图的边框线和胡须的线宽度，fill设置填充颜色，outlier.fill和outlier.color设置异常点的属性
  #geom_jitter(aes(fill=variable),width =0.2,shape = 21,size=2.5)+ #设置为向水平方向抖动的散点图，width指定了向水平方向抖动，不改变纵轴的值
  #geom_point(aes(colour = factor(canshu),shape=c(21:25)))+
  geom_point(aes(size = 0.1,shape=canshu,fill=canshu)) +
  scale_shape_manual(values = c(21:25)) +
  scale_fill_manual(values = pal(5))+
  
  #scale_fill_manual(values = pal(12))+  #设置填充的颜色
  #scale_color_manual(values=c("black","black","black"))+ #设置散点图的圆圈的颜色为黑色
  ggtitle("ACCURACY")+ #设置总的标题
  theme_bw()+ #背景变为白色
  theme(
    #legend.position="none", #不需要图例
    axis.text.x=element_text(colour="black",family="Times",size=14), #设置x轴刻度标签的字体属性
    axis.text.y=element_text(family="Times",size=14,face="plain"), #设置x轴刻度标签的字体属性
    axis.title.y=element_text(family="Times",size = 14,face="plain"), #设置y轴的标题的字体属性
    axis.title.x=element_text(family="Times",size = 14,face="plain"), #设置x轴的标题的字体属性
    plot.title = element_text(family="Times",size=15,face="bold",hjust = 0.5), #设置总标题的字体属性
    panel.grid.major = element_blank(), #不显示网格线
    panel.grid.minor = element_blank())+
  ylab("ACCURACY")+xlab("METHODS") +#设置x轴和y轴的标题)
  theme(axis.text.x = element_text(angle = 30,vjust = 0.5) )      # x轴刻度改为倾斜90度，防止名称重叠



P1

pdf(pathout,height = 6,width =5)

ggplot(df,aes(x=variable,y=value,fill=canshu))+ #'fill='设置填充颜色
  #stat_boxplot(geom = "errorbar",width=0.15)+ #由于自带的箱形图没有胡须末端没有短横线，使用误差条的方式补上
  geom_boxplot(alpha = 0.6,size=1,fill="white",outlier.color = "white",outlier.fill="white")+ #size设置箱线图的边框线和胡须的线宽度，fill设置填充颜色，outlier.fill和outlier.color设置异常点的属性
  #geom_boxplot(size=0.5,fill="white",outlier.fill="white",outlier.color="white")+ #size设置箱线图的边框线和胡须的线宽度，fill设置填充颜色，outlier.fill和outlier.color设置异常点的属性
  #geom_jitter(aes(fill=variable),width =0.2,shape = 21,size=2.5)+ #设置为向水平方向抖动的散点图，width指定了向水平方向抖动，不改变纵轴的值
  #geom_point(aes(colour = factor(canshu),shape=c(21:25)))+
  #geom_point(aes(colour = factor(canshu),shape=canshu),fill="gray",size=2) +
  geom_jitter(aes(colour = factor(canshu),shape=canshu),fill="gray",width =0.2,size=2.5)+ 
  
  scale_shape_manual(values = c(21:25)) +
  #scale_shape_manual(values = c(0,1,2,5,6)) +
  scale_color_manual(values=c("#FF0000","#008B45","#0000FF","#FFFF00","#A020F0"))+ #设置散点图的圆圈的颜色为黑色
  scale_fill_manual(values = 'gray')+
  #设置为向水平方向抖动的散点图，width指定了向水平方向抖动，不改变纵轴的值
  
  #scale_fill_manual(values = pal(12))+  #设置填充的颜色
  #scale_color_manual(values=c("black","black","black"))+ #设置散点图的圆圈的颜色为黑色
  ggtitle("Pancreas")+ #设置总的标题
  theme_bw()+ #背景变为白色
  theme(
    #legend.position="none", #不需要图例
    axis.text.x=element_text(colour="black",family="Times",size=14), #设置x轴刻度标签的字体属性
    axis.text.y=element_text(family="Times",size=14,face="plain"), #设置x轴刻度标签的字体属性
    axis.title.y=element_text(family="Times",size = 14,face="plain"), #设置y轴的标题的字体属性
    axis.title.x=element_text(family="Times",size = 14,face="plain"), #设置x轴的标题的字体属性
    plot.title = element_text(family="Times",size=15,face="bold",hjust = 0.5), #设置总标题的字体属性
    panel.grid.major = element_blank(), #不显示网格线
    panel.grid.minor = element_blank())+
  ylab("Score")+xlab("METHODS") +#设置x轴和y轴的标题)
  theme(axis.text.x = element_text(angle = 30,vjust = 0.5) )      # x轴刻度改为倾斜90度，防止名称重叠

dev.off()


ggplot(df, aes(x = canshu, y = value,color = canshu)) +
  geom_beeswarm(cex = 2,size=2.5, priority = "descending") +
  scale_color_brewer(palette = "Dark2")+
  geom_jitter(aes(colour = factor(canshu)),fill="gray",width =0.2,size=2.5)+ 
  coord_cartesian(ylim = c(0, 1))+
  theme_classic()+
  stat_summary(fun.y = median, fun.ymin = median, fun.ymax = median, #median或mean
               geom = 'crossbar', width = 0.5, size = 0.3, color = 'black')+
  stat_summary(fun.data = function(x) median_hilow(x, 0.5), 
               geom = 'errorbar', width = 0.2, color = 'black')+
  ylab("Score")+#设置x轴和y轴的标题)
  theme(axis.text.x = element_text(angle = 30,vjust = 0.5))










