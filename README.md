# Krypton Geo Tagger

finds geographic locations in text

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Docker and Docker Compose

### DB Deployment

You need a running Mongo instance:

```
docker-compose  up -d mongo
```

And loading the geographic collections of the city, in our case Montevideo.

```
docker-compose  up -d restore
```

You can optionally run Mongo Express to explore the data:

```
docker-compose  up -d mongo-express
```

### Tagger Deployment

You need to run the service in an exposed port, default is 5000

```
docker-compose up -d tagger
```


## Running

Krypton Geo Tagger is an async service available as a Rest endpoint. To find locations in the text: "*contenedor sucio en pocitos concretamente charrúa y pablo de maría*", you need to issue a request with a JSON payload. You will get the results as a POST request in the endpoint specified in the "callback" field. 

**Find Locations**
----
  Finds locations asynchronously.

* **URL**

  /api/find/:id

* **Method:**

  `POST`
  
*  **URL Params**

   **Required:**
 
   `id=[integer]`

* **Data Params**

   `{
		callback : "http://localhost:5000/api/print",
	    text : "contenedor sucio en pocitos concretamente charrúa y pablo de maría"
}`

* **Success Response:**

  * **Code:** 200 <br />
    **Content:** `OK`

**Log results**
----
  Logs results.

* **URL**

  /api/print/

* **Method:**

  `POST`
  
* **Data Params**

   `{}`

* **Success Response:**

  * **Code:** 200 <br />
    **Content:** `OK`
 




### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Results

The JSON payload structure of the results for request id=1 is as follows:

```json
{
   'error':False,
   'id':'1',
   'solutions':[
      {
         'centroid':[
            [
               -56.17093567532716,
               -34.90616306208891
            ]
         ],
         'elements':[

         ],
         'score':{
            'combined':18.666666666666664,
            'count':2,
            'mongo':2.333333333333333,
            'ngram':4
         }
      }
   ]
}
```
The list of solutions will be ordered by score. Each solution has a centroid field wich is a representative point of the solution and a list of elements that make up the solution.



## Contributing

You are free to fork and PR with improvements.


## Authors

* **Raúl Speroni** - [msteglichc@gmail.com](https://github.com/martin-steglich)
* **Martín Steglich** - [raulsperoni@gmail.com](https://github.com/raulsperoni)