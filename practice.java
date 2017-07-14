import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.Writer;
import java.time.LocalDateTime;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class practice2 {
	public static void main(String[] args) {
		//System.err.close();
		
		//System.out.println(LocalDateTime.now());
		anomalies da = new anomalies("C:\\Users\\ZK0GJXO\\Documents\\Files.txt");
		da.printAnomalies();
		
		/*try {
			PrintWriter writer = new PrintWriter("C:\\Users\\ZK0GJXO\\Documents\\Files.txt");
			writer.println("File\tSize\tVolume");
			for(int i = 1; i <= 100; i++){
				double random = Math.random() + 1;
				int size = (int) (50 * random);
				int records = (int) (1500 * random);
				
				writer.println("F" + i + "\t" + size + "\t" + records);
			}
			
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		
		/*SparkConf sparkConf = 
				new SparkConf()
				.setAppName("Anomaly Detection")
				.setMaster("local");
		
		JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
		
		SparkSession sparkSession = 
				SparkSession
				  .builder()
				  .appName("Anomaly Detection")
				  .getOrCreate();
		
		Dataset<Row> dataFrame = sparkSession.read().
				option("header", "true").
				option("delimiter", "\t").
				option("inferSchema", "true").
				csv("C:\\Users\\ZK0GJXO\\Documents\\Files.txt");  //reads the data from a file whose path is passed to the constructor 
		
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"Size", "Records"})
				.setOutputCol("[Size, Records]");
		
		dataFrame = assembler.transform(dataFrame);
		
		StandardScaler normalizer = new StandardScaler()
				.setWithMean(true)
				.setWithStd(true)
				.setInputCol("[Size, Records]")
				.setOutputCol("features");
		
		dataFrame = normalizer.fit(dataFrame).transform(dataFrame);
		
		GaussianMixture gm = new GaussianMixture();
		GaussianMixtureModel gmm = gm.fit(dataFrame);
		
		gmm.transform(dataFrame).show(false);
		//System.out.println(gmm.summary().clusterSizes()[0]);

		//System.out.println(gmm.predictProbability(dataFrame.collectAsList().get(0).getAs("features")));*/
		
	}
}
