library(Seurat)
library(tidyverse)
### Rscript External_test_group_preprocess.R input_path reference query out_path
args <- commandArgs()
path= args[6]
path_rf_c=paste0(path,args[7],'/',args[7],'_count.csv')
path_rf_type=paste0(path,args[7],'/',args[7],'_type.csv')
rf_c=read.csv(path_rf_c,header=T,row.names=1)
rf_type=read.csv(path_rf_type,header=T,row.names=1)
colnames(rf_type)<-c('cell_type')
rf_type$name<-rownames(rf_type)
#rf_c=read.csv("/data/yrj/download_sc/Pancreas/Baron/baron_count.csv",header=T,row.names=1)
#rf_type=read.csv("/data/yrj/download_sc/Pancreas/Baron/baron_type.csv",header=T,row.names=1)

path_query_c=paste0(path,args[8],'/',args[8],'_count.csv')
path_query_type=paste0( path,args[8],'/',args[8],'_type.csv')

query_c=read.csv(path_query_c,header=T,row.names=1)
query_type=read.csv(path_query_type,header=T,row.names=1)
#query_c=read.csv("/data/yrj/download_sc/Pancreas/Mu_Se_vs_Wa/muraro_count.csv",header=T,row.names=1)
#query_all=read.csv("/data/yrj/download_sc/Pancreas/Mu_Se_vs_Wa/muraro_data.csv",header=T,row.names=1)
#query_type<-query_all[dim(query_all)[1],]

#rf_c[is.na(rf_c)] <- 0

rf_type[rf_type==c('pp')]<-c('gamma')
rf_type[rf_type==c('duct')]<-c('ductal')
query_type[query_type==c('pp')]<-c('gamma')
query_type[query_type==c('duct')]<-c('ductal')
#rf_type[rf_type==c('activated_stellate')]<-c('PSC')
#rf_type[rf_type==c('quiescent_stellate')]<-c('PSC')
rf_type[str_which(rf_type$cell_type,"_stellate"),]<-c('PSC')

rf_type[str_which(rf_type$cell_type,".contaminated"),]<-c('dropped')
#rf_type<-rf_type[rf_type!='dropped']
#rf_type2<-rf_type[rf_type!='beta.contaminated'&rf_type!='alpha.contaminated']
#query_type[str_which(query_type,"_stellate"),]<-c('PSC')
rf_type2<-rf_type[rf_type$cell_type!='dropped',]

rf_c2<-rf_c[,colnames(rf_c)%in%rf_type2$name]

rf_type3<-rf_type2[colnames(rf_c2),]

##删基因
rf_del<-rf_c2[rowSums(rf_c2)>2,]

##order marker

order_marker<-read.csv('~/download_sc/Pancreas/pancreas_marker_order_gene.csv',header = T,row.names = 1)
rf_marker_c<-rf_del[order_marker$gene_name,]
rf_marker_c<-na.omit(rf_marker_c)

query_marker_c<-query_c[rownames(rf_marker_c),]

rownames(query_marker_c)<-rownames(rf_marker_c)
query_marker_c[is.na(query_marker_c)] <- 0



rf <- CreateSeuratObject(counts = rf_marker_c, project = "Baron", min.cells = 0)
#Baron <- subset(Baron, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
rf <- NormalizeData(rf, normalization.method = "LogNormalize", scale.factor = 10000)
all.genes <- rownames(rf)
rf <- ScaleData(rf, features = all.genes)



query <- CreateSeuratObject(counts = query_marker_c, project = "muraro", min.cells = 0)
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
  
  rf_type_count<-table(rf_type$cell_type)
  delet_type=rf_type_count[rf_type_count< num]
  names(delet_type)
  if (length(delet_type) < 1 ){
    
    return(rf_type)
    
  }else {
    
    test_type=rf_type$cell_type!=names(delet_type)[1]
    
    for (i in 1:length(delet_type)){
      
      test_type=test_type&( rf_type$cell_type!=names(delet_type)[i])
      
    }
    
    rf_type_left=rf_type$cell_type[test_type]
    #rf_type_left=query_type$cell_type[test_q_type]
    rf_type_left<-as.data.frame(rf_type_left)
    rownames(rf_type_left)<-rownames(rf_type)[test_type]
    
    return(rf_type_left)
    
    
  }
  
  
}

rf_type_left<-delet_cell_type(rf_type3,num=10)
colnames(rf_type_left)<-c('cell_type')
print(dim(rf_type_left))
#rf_type_left_name<-unique(rf_type_left$rf_type_left)
#unique(rf_type_left)[i] %in%

###手动 cell_type 统一


#rf_type_left[str_which(rf_type_left,"cinar")]<-c('acinar')
#rf_type_left[str_which(rf_type_left,"eta")]<-c('beta')
#rf_type_left[str_which(rf_type_left,"elta")]<-c('delta')
#rf_type_left[str_which(rf_type_left,"stellate")]<-c('PSC')
#rf_type_left[str_which(rf_type_left,"uct")]<-c('ductal')
#rf_type_left[str_which(rf_type_left,"lpha")]<-c('alpha')
#rf_type_left[str_which(rf_type_left,"psilon")]<-c('epsilon')
#rf_type_left[str_which(rf_type_left,"amma")]<-c('gamma')
#rf_type_left[str_which(rf_type_left,"ndothelial")]<-c('endothelial')
#rf_type_left[str_which(rf_type_left,"acrophage")]<-c('macrophage')
#rf_type_left[str_which(rf_type_left,"chwann")]<-c('schwann')


#query_type_left[str_which(query_type_left,"cinar")]<-unique(rf_type_left[str_which(rf_type_left,"cinar")])






###循环实现上述替换
#left_chars<- substr(chars, 2, nchar(chars))

#rf_type_left[str_which(rf_type_left,left_chars[1])]<-c('acinar')


###### 挑选query_type在rf_type中的cell type

select_query_type<-function(query_type,rf_type_left)
{
  query_type_count<-table(query_type$cell_type)
  query_panduan = as.data.frame(matrix(ncol=1,nrow=length(unique(query_type_count))))
  
  
  for (i in 1:length(unique(query_type_count))) {
    
    query_panduan[i,]<-names(query_type_count)[i] %in% unique(rf_type_left$cell_type)
    
    
  }
  #query_type_count[query_panduan$V1]
  #query_type_left=query_type$cell_type[names(query_type_count[query_panduan$V1])]
  left_query_name<-names(query_type_count[query_panduan$V1]) 
  
  test_q_type=query_type$cell_type==left_query_name[1]
  
  for (i in 1:length(left_query_name)){
    
    test_q_type=test_q_type|(query_type$cell_type==left_query_name[i])
    
  }
  #test_q_type<-as.data.frame(test_q_type)
  #rownames(test_q_type)<-rownames(query_type)
  query_type_left=query_type$cell_type[test_q_type]
  query_type_left<-as.data.frame(query_type_left)
  rownames(query_type_left)<-rownames(query_type)[test_q_type]
  
  return(query_type_left)
}

#rf_type_left2<-rf_type_left$cell_type
query_type_left<-select_query_type(query_type,rf_type_left)
###手动 cell_type 统一

#rf_type_left[str_which(rf_type_left,"cinar")]<-c('acinar')

#### cell type 和 counts的统一

rf_c_left<-rf_c[,rownames(rf_type_left)]
query_c_left<-query_c[,rownames(query_type_left)]


colnames(rf_type_left)<-c('cell_type')
rf_all<-rbind(rf_c_left,t(rf_type_left$cell_type))
query_all<-rbind(query_c_left,t(query_type_left))

print(dim(rf_all))
print(dim(query_all))


#query_data_mapped<-rbind(query_c,query_type)
#rf_del_all<-rbind(rf_del,rf_type$cell_type1)
path_out_rf= paste0(args[9],args[7],'_all.csv')
path_out_query= paste0(args[9],args[8],'_all.csv')


write.csv(rf_all,path_out_rf)
write.csv(query_all,path_out_query)
