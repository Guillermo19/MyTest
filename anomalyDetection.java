public class DetectAnomalies {
	private BisectingKMeans kmeans; 
	private BisectingKMeansModel model;
	
	private Dataset<Row> dataFrame;
	private Dataset<Row> vectorizedDataFrame;
	
	private Vector[] clusterCenters;
	
	private final static SparkConf sparkConf = 
			new SparkConf()
			.setAppName("Anomaly Detection")
			.setMaster("local");
	
	private final static JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
	
	private final static SparkSession sparkSession = 
			SparkSession
			  .builder()
			  .appName("Anomaly Detection")
			  .getOrCreate();
	
	
	public DetectAnomalies(String filePath) {
	
		dataFrame = sparkSession.read().
				option("header", "true").
				option("delimiter", "\t").
				option("inferSchema", "true").
				csv(filePath);  //reads the data from a file whose path is passed to the constructor 
		
		dataFrame = dataFrame.withColumn("Size", dataFrame.col("Size").cast("integer")); // casts both the "size" and "records" column as integer types 
		dataFrame = dataFrame.withColumn("Records", dataFrame.col("Records").cast("integer"));
		
		dataFrame.persist(); //persists (cache) the data set on memory 
		
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"Size", "Records"})
				.setOutputCol("features");
		
		vectorizedDataFrame = assembler.transform(dataFrame).persist(); //converts and combines the 'size' and 'records' column as the vector type 'features' column
																		//this is the column k-means works with
		kmeans = new BisectingKMeans().setSeed(1l).setK(6);
		
		model = kmeans.fit(vectorizedDataFrame); //trains the k-means with the data set to analyze to create a model
		
		clusterCenters = model.clusterCenters(); //get the cluster centers for the model just created
	
	}
	
	public void printAnomalies() {
		
		Column sizeCol = this.dataFrame.col("Size");
		Column recordsCol = this.dataFrame.col("Records");
		
		List<Double> clusterSize = new ArrayList<>(); //list that will hold the size part of a cluster center (x coord)
		List<Double> clusterRecords = new ArrayList<>(); //list that will hold the records part of a cluster center (y coord)
		
		Dataset<Row> filteredDataSet;
		
		for(Vector cluster : this.clusterCenters) { //loop that appends the cluster centers to the 'size' and 'records' arrays. 
			clusterSize.add(cluster.toArray()[0]); 
			clusterRecords.add(cluster.toArray()[1]);
		}
		
		filteredDataSet = this.dataFrame //filters the data set by getting the rows whose 'size' column values are in the 'clusterSize' list and 'records' column values are in the 'clusterRecords' list
				.where(sizeCol.isin(clusterSize.toArray())
				.and(recordsCol.isin(clusterRecords.toArray()))
				);
		
		filteredDataSet.show(); //shows the filters data set. this is supposed to show the outliers / anomalous points 
		
	}
 }
