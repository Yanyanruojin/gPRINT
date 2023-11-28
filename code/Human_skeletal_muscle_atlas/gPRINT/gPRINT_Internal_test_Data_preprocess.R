args<-commandArgs(T)

pathway<-c('/data/Human_skeletal_muscleatlas/')

path1<-paste0(pathway,args[1],"/",args[1],"_count.csv")
path2<-paste0(pathway,args[1],"/",args[1],"_type.csv")
outway<-paste0(pathway,args[1],"/cnn_self/")
path3<-paste0(outway,args[1],"_order_all.csv")
print(paste(path1,path2,path3,seq=" "))
data<-read.csv(path1,header=T,row.names=1)
data_celltype<-read.csv(path2,header=T,row.names=1)
ordergenes<-read.csv("/data/yrj/download_sc/order_end_gene.csv",header=T,row.names=1)

data_order_gene<-ordergenes[ordergenes$gene_id%in%rownames(data),]
data_order<-data[data_order_gene$gene_id,]

#delet

delet_2<- function(df){ 
  df_sum<-rowSums(df)
  df_delet<-df[names(df_sum)[df_sum>2],]
  return(df_delet) 
}


##merge
zhenghe<-function(df){
  df_delet<-delet_2(df)
  data_celltype<-data_celltype[colnames(df_delet),]
  cell_type<-as.data.frame(t(data_celltype))
  colnames(cell_type)<-colnames(df_delet)
  #cell_type<-cell_type[-1,]
  data_all<-rbind(df_delet,cell_type)
  return(data_all)
}

data_all<-zhenghe(data_order)
write.csv(data_all,path3)
