### Rscript gPRINT_External_test_Data_preprocess.R train query 
###


args<-commandArgs(T)

pathway<-c('/data/Human_skeletal_muscle_atlas/')

path_rf_c<-paste0(pathway,args[1],"/",args[1],"_count.csv")
path_rf_type<-paste0(pathway,args[1],"/",args[1],"_type.csv")

rf_c=read.csv(path_rf_c,header=T,row.names=1)
rf_type=read.csv(path_rf_type,header=T,row.names=1)

rf_type2<-as.data.frame(rf_type)

rownames(rf_type2)<-rownames(rf_type)
colnames(rf_type2)<-c("celltype")

rf_type<-rf_type2


pathway2<-c('/data/yrj/MTJ/人类骨骼肌图谱/')

path_query_c=paste0(pathway2,args[2],"/",args[2],"_count.csv")
path_query_type=paste0(pathway2,args[2],"/",args[2],"_type.csv")

query_c=read.csv(path_query_c,header=T,row.names=1)
query_type=read.csv(path_query_type,header=T,row.names=1)
query_type2<-as.data.frame(query_type)

rownames(query_type2)<-rownames(query_type)
colnames(query_type2)<-c("celltype")
query_type<-query_type2


#query_c=read.csv("/data/yrj/download_sc/Pancreas/Mu_Se_vs_Wa/muraro_count.csv",header=T,row.names=1)
#query_all=read.csv("/data/yrj/download_sc/Pancreas/Mu_Se_vs_Wa/muraro_data.csv",header=T,row.names=1)
#query_type<-query_all[dim(query_all)[1],]

#rf_c[is.na(rf_c)] <- 0
rf_c2<-rf_c[,colnames(rf_c)%in%rownames(rf_type)]
rf_type3<-rf_type[colnames(rf_c2),]
rf_type3<-as.data.frame(rf_type3)
colnames(rf_type3)<-c("celltype")

rownames(rf_type3)<-colnames(rf_c2)
rf_del<-rf_c2[rowSums(rf_c2)>2,]


##ordergenes
ordergenes<-read.csv("/data/yrj/download_sc/order_end_gene.csv",header=T,row.names=1)
ordergenes<-ordergenes[!duplicated(ordergenes$gene_id),]

#ordergenes$gene_id[ordergenes$gene_id %in% rownames(rf_del)]
rf_del2<-rf_del[ordergenes$gene_id[ordergenes$gene_id %in% rownames(rf_del)],]

rf_del2<-na.omit(rf_del2)
query_c<-query_c[rownames(rf_del2),]
rownames(query_c)<-rownames(rf_del2)
query_c[is.na(query_c)] <- 0




rf_c<-rf_del2
#ba_marker_type2<-t(ba_marker_type)
print(dim(rf_c))
print(dim(query_c))

rf_type_need<-t(rf_type3$celltype)
colnames(rf_type_need)<-rownames(rf_type3)
rf_all<-rbind(rf_c,rf_type_need)


query_type2<-query_type[colnames(query_c),]
query_all<-rbind(query_c,query_type2)

#print(dim(rf_all))
print(dim(query_all))



#query_data_mapped<-rbind(query_c,query_type)
#rf_del_all<-rbind(rf_del,rf_type$cell_type1)
#path_out_rf= paste0(args[9],args[7],'_all.csv')
path_out_rf= paste0('/data/Human_skeletal_muscle_atlas/Wk_12vs17/',
                    args[1],'_train_all.csv')

path_out_query= paste0('/data/Human_skeletal_muscle_atlas/Wk_12vs17/',
                       args[1],'_vs_',args[2],'_all.csv')


write.csv(rf_all,path_out_rf)
write.csv(query_all,path_out_query)



  
