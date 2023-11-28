# find markers for every cluster compared to all remaining cells, report only the positive
# ones

Idents(train)<-train$ordinal
train.markers <- FindAllMarkers(train, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 1.5)

Idents(test)<-test$ordinal
test.markers <- FindAllMarkers(test, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 1.5)

write.csv(train.markers,'./Wk_12_14/marker/train_marker.csv')
write.csv(test.markers,'./Wk_12_14/marker/test_marker.csv')

train_marker_c<-trainMatrix[train.markers$gene,]
train_marker_type<-trainMetaData[colnames(train_marker_c),]

##order gene
ordergenes<-read.csv("/data/order_end_gene.csv",header=T,row.names=1)
ordergenes<-ordergenes[!duplicated(ordergenes$gene_id),]

#ordergenes$gene_id[ordergenes$gene_id %in% rownames(rf_del)]
train_marker_order<-train_marker_c[ordergenes$gene_id[ordergenes$gene_id %in% rownames(train_marker_c)],]

train_marker_order<-na.omit(train_marker_order)


test.markers<-as.data.frame(test.markers)
name_need<-rownames(test.markers)[rownames(test.markers)%in%rownames(train_marker_order)]
test.markers2<-test.markers[name_need,]
train_marker_order2<-train_marker_order[name_need,]



test_marker_c<-testMatrix[rownames(test.markers2),]
test_marker_type<-testMetaData[colnames(test_marker_c),]





rf_all<-rbind(train_marker_order2,train_marker_type)


query_all<-rbind(test_marker_c,test_marker_type)

#print(dim(rf_all))
print(dim(query_all))



#query_data_mapped<-rbind(query_c,query_type)
#rf_del_all<-rbind(rf_del,rf_type$cell_type1)
#path_out_rf= paste0(args[9],args[7],'_all.csv')
path_out_rf= paste0('/data/Human_skeletal_muscle_atlas/Wk_12_14/marker/train_marker2_all.csv')

path_out_query= paste0('/data/Human_skeletal_muscle_atlas/Wk_12_14/marker/test_marker2_all.csv')



write.csv(rf_all,path_out_rf)
write.csv(query_all,path_out_query)






