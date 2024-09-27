

### gprint_pre_reference.R $1 $2
### gprint_pre_reference.R human_Pancreas349 Pancreas

args<-commandArgs(T)

setwd("..")


path_main = './reference_model/'

pathe_gene = paste0(path_main,args[2],'/model/cell_level/order_genes_reference.csv')


gene_names<-read.csv(pathe_gene,header = F)
colnames(gene_names)<-"gene_name"


path_query<-c('./data/')

path_query_c=paste0(path_query,"/",args[1],"_count.csv")
path_query_type=paste0(path_query,"/",args[1],"_celltype.csv")

query_c=read.csv(path_query_c,header=T,row.names=1)
query_type=read.csv(path_query_type,header=T,row.names=1)
colnames(query_type)<-'celltype'

query_c<-query_c[gene_names$gene_name,]
rownames(query_c)<-gene_names$gene_name
query_c[is.na(query_c)] <- 0
query_type2<-query_type[colnames(query_c),,drop = FALSE]
query_all<-rbind(query_c,query_type2$celltype)

#print(dim(rf_all))
print(dim(query_all))

path_out_query= paste0(path_main,args[2],'/query/',args[1],'_query_all.csv')
write.csv(query_all,path_out_query)










