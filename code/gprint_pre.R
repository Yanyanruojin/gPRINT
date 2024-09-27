library(Seurat)
library(tidyverse)

## args[1]: reference  ;  args[2]: query; 
args<-commandArgs(T)

### create folder

setwd("..")


folder_name <- paste0('./match_model/',args[1],'_anno_',args[2],'/')

if (!dir.exists(folder_name)) {
  dir.create(folder_name)
  cat("Folder created successfully：", folder_name, "\n")
} else {
  cat("The folder already exists：", folder_name, "\n")
}


folder_name1 = paste0(folder_name,'model/cell_level/')
folder_name2 = paste0(folder_name,'result/cell_level/')
folder_name3 = paste0(folder_name,'query/')
folder_name4 = paste0(folder_name,'reference/')


dir.create(folder_name1)
dir.create(folder_name2)
dir.create(folder_name3)
dir.create(folder_name4)


#### reference 

outway1 <- paste0(folder_name4,args[1],"_all.csv")

pathway<-c('./data/')

path_rf_c<-paste0(pathway,args[1],"_count.csv")
path_rf_type<-paste0(pathway,args[1],"_celltype.csv")

rf_c=read.csv(path_rf_c,header=T,row.names=1)
rf_type=read.csv(path_rf_type,header=T,row.names=1)

colnames(rf_type)<-'celltype'

#rf_c=read.csv("/data/yrj/download_sc/Pancreas/Baron/baron_count.csv",header=T,row.names=1)
#rf_type=read.csv("/data/yrj/download_sc/Pancreas/Baron/baron_type.csv",header=T,row.names=1)

path_query_c=paste0(pathway,args[2],"_count.csv")
path_query_type=paste0(pathway,args[2],"_celltype.csv")

query_c=read.csv(path_query_c,header=T,row.names=1)
query_type=read.csv(path_query_type,header=T,row.names=1)
colnames(query_type)<-'celltype'

# 获取交集
intersection_cell <- intersect(colnames(query_c), rownames(query_type))


query_c <- query_c[,intersection_cell]
query_type <- query_type[intersection_cell, , drop = FALSE]





rf_c2<-rf_c[,colnames(rf_c)%in%rownames(rf_type)]

rf_type3<-rf_type[colnames(rf_c2), , drop = FALSE]

rf_del<-rf_c2[rowSums(rf_c2)>2,]


##ordergenes
ordergenes<-read.csv("./data/order_end_gene.csv",header=T,row.names=1)
ordergenes<-ordergenes[!duplicated(ordergenes$gene_id),]
#ordergenes$gene_id[ordergenes$gene_id %in% rownames(rf_del)]
rf_del2<-rf_del[ordergenes$gene_id[ordergenes$gene_id %in% rownames(rf_del)],]

rf_del2<-na.omit(rf_del2)
query_c<-query_c[rownames(rf_del2),]
rownames(query_c)<-rownames(rf_del2)
query_c[is.na(query_c)] <- 0



rf <- CreateSeuratObject(counts = rf_del2, min.cells = 0)
#Baron <- subset(Baron, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
rf <- NormalizeData(rf, normalization.method = "LogNormalize", scale.factor = 10000)
all.genes <- rownames(rf)
rf <- ScaleData(rf, features = all.genes)



query <- CreateSeuratObject(counts = query_c,  min.cells = 0)
query <- NormalizeData(query, normalization.method = "LogNormalize", scale.factor = 10000)
all.genes <- rownames(query)
query <- ScaleData(query, features = all.genes)




rf_c<-as.matrix(GetAssayData(rf))

query_c<-as.matrix(GetAssayData(query))
#ba_marker_type2<-t(ba_marker_type)
print(dim(rf_c))
print(dim(query_c))


###删除细胞数小于10的细胞类型

delet_cell_type<-function(rf_type,num){
  
  rf_type_count<-table(rf_type$celltype)
  delet_type=rf_type_count[rf_type_count< num]
  names(delet_type)
  if (length(delet_type) < 1 ){
    
    return(rf_type)
    
  }else {
    
    test_type=rf_type$celltype!=names(delet_type)[1]
    
    for (i in 1:length(delet_type)){
      
      test_type=test_type&( rf_type$celltype!=names(delet_type)[i])
      
    }
    
    
    rf_type_left=rf_type$celltype[test_type]
    #rf_type_left=query_type$cell_type[test_q_type]
    rf_type_left<-as.data.frame(rf_type_left)
    rownames(rf_type_left)<-rownames(rf_type)[test_type]
    
    return(rf_type_left)
    
    
  }
  
  
}

rf_type_left<-delet_cell_type(rf_type3,num=10)

print(dim(rf_type_left))

#rf_type_left[str_which(rf_type_left,"cinar")]<-c('acinar')

#### cell type 和 counts的统一

rf_c_left<-rf_c[,rownames(rf_type_left)]
#rownames(query_type)<-gsub("-",".",rownames(query_type))
query_c_left<-query_c[,rownames(query_type)]


rf_all<-rbind(rf_c_left,t(rf_type_left$celltype))
query_all<-rbind(query_c_left,t(query_type$celltype))

print(dim(rf_all))
print(dim(query_all))



#query_data_mapped<-rbind(query_c,query_type)
#rf_del_all<-rbind(rf_del,rf_type$cell_type1)

path_out_rf= paste0(folder_name4,args[1],'_all.csv')
path_out_query= paste0(folder_name3,args[2],'_all.csv')


write.csv(rf_all,path_out_rf)
write.csv(query_all,path_out_query)



