import java.util.ArrayList;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class DetectAnomalies {
	private BisectingKMeans kmeans; 
	private BisectingKMeansModel model;
	
	private Dataset<Row> dataset;
	private Dataset<Row> normalizedDataset;
	private Dataset<Row> vectorDataset;
	
	private Vector[] clusterCenters;
	private ArrayList<Row> anomalies;
	
	private StandardScaler normalizer;
	
	private final static double threshold = 2.5;
	
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
	
		dataset = sparkSession.read().
				option("header", "true").
				option("delimiter", "\t").
				option("inferSchema", "true").
				csv(filePath);  //reads the data from a file whose path is passed to the constructor 
		
		dataset.cache(); 
		
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"Size", "Volume"})
				.setOutputCol("Inputs");
		
		vectorDataset = assembler.transform(dataset); //converts and combines the 'size' and 'records' column as the vector type 'features' column
		vectorDataset.cache();
													 
		normalizer = new StandardScaler()             //The normalizer standardizes the given input columns to make it fit a normal distribution. 
					.setWithMean(true)				 //It standardizes using the mean and standard deviation
					.setWithStd(true)
					.setInputCol("Inputs")
					.setOutputCol("features");
		
		normalizedDataset = normalizer.fit(vectorDataset).transform(vectorDataset); //creates the normalized dataset by standardizing the vectorDataset
		normalizedDataset.cache(); //saves it in memory for later use 
		
		kmeans = new BisectingKMeans();
		
		model = kmeans.fit(normalizedDataset); //trains the k-means with the dataset to create a model
		
		clusterCenters = model.clusterCenters(); 
		
		dataset.show(false);
		
		for(Vector v : clusterCenters){
			System.out.println(v);
		}
		this.adjustModel();
	} //end constructor
	
	private void adjustModel(){ //Identifies potential extreme outliers (outliers far away from the normal data), saves and removes them, and adjusts the model with the new dataset 
		
		anomalies = new ArrayList<>(); 

		Dataset<Row> transDataset = model.transform(normalizedDataset); //creates a temporary transformed normalizedDataset that contains a prediction column
		transDataset.cache(); 
		Column predictionCol = new Column("prediction");
		
		for(Vector v : clusterCenters){
			double magnitude = Math.sqrt(Vectors.sqdist(Vectors.zeros(v.size()), v)); //finds the magnitude of each cluster center, which is the distance from the origin to it's position
			
			if(magnitude > threshold) {
				
				anomalies.addAll(transDataset.filter(predictionCol.equalTo(model.predict(v))).collectAsList()); 
				
				transDataset = transDataset.filter(predictionCol.notEqual(model.predict(v)));
				
			}//end if
		}//end for
		
		normalizedDataset = normalizer.fit(transDataset.drop("prediction", "features")).transform(transDataset.drop("prediction", "features"));
		model = kmeans.fit(normalizedDataset);
		clusterCenters = model.clusterCenters();
		
		this.getAnomalies();
		
	}//end adjust model
	
	private void getAnomalies(){ //identifies potential outliers that might be mixed with normal data and saves them
		ArrayList<Row> rows = new ArrayList<>(normalizedDataset.collectAsList());
		int i = 0;
		Vector datapoint;
		Vector cluster;
		Column fileName = new Column("File");
		Dataset<Row> temp = normalizedDataset.drop("features").cache();
		int MaxOutlierIndex = -1;
		double MaxDistanceFromCluster = 0;
		
		while(i < rows.size()){
			datapoint = rows.get(i).getAs("features");
			cluster = clusterCenters[model.predict(datapoint)];
			double distance = Math.sqrt(Vectors.sqdist(datapoint, cluster));
			
			if(distance > threshold && distance > MaxDistanceFromCluster){
				MaxDistanceFromCluster = distance;
				MaxOutlierIndex = i;
			}
			
			else
				i++;
			
			if(i == rows.size() - 1 && MaxOutlierIndex > -1 && MaxDistanceFromCluster > 0){
				temp = temp.filter(fileName.notEqual(rows.get(MaxOutlierIndex).getAs("File")));
				normalizedDataset = normalizer.fit(temp).transform(temp);
				model = kmeans.fit(normalizedDataset);
				clusterCenters = model.clusterCenters();
	
				anomalies.add(rows.get(MaxOutlierIndex));
				rows = new ArrayList<>(normalizedDataset.collectAsList());
				i = 0;
				MaxOutlierIndex = -1;
				MaxDistanceFromCluster = 0;
			}
		}
	}//end get anomalies
	
	public void printAnomalies() {
		if(!anomalies.isEmpty()){
			System.out.println("Possible anomalous files: ");
			for(Row r : anomalies){
				System.out.println(r.getAs("File").toString());
			}
		}
		else{
			System.out.println("No probable anomalous files");
		}
		
	} //end print anomalies 
}//end class
