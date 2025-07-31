import faiss
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Attraction:
    """Data structure for travel attractions"""
    id: str
    city: str
    country: str
    name: str
    description: str
    category: str
    avg_cost_usd: float
    rating: float
    latitude: float
    longitude: float
    address: str
    opening_hours: str
    website: str
    phone: str
    tags: List[str]
    image_url: str
    created_at: str
    last_updated: str

class TravelVectorDB:
    """
    Vector database for travel attractions using Faiss
    Supports similarity search and CRUD operations
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', dimension: int = 384):
        """
        Initialize the vector database
        
        Args:
            model_name: Sentence transformer model name
            dimension: Vector dimension (384 for MiniLM-L6-v2)
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.attractions: List[Attraction] = []
        self.id_to_idx: Dict[str, int] = {}
        
        logger.info(f"Initialized TravelVectorDB with model: {model_name}")
    
    def generate_realistic_data(self, count: int = 200) -> List[Attraction]:
        """
        Generate realistic attraction data using free APIs
        
        Args:
            count: Number of attractions to generate
            
        Returns:
            List of Attraction objects
        """
        attractions = []
        
        cities = [
            {"name": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
            {"name": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
            {"name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
            {"name": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
            {"name": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964},
            {"name": "Barcelona", "country": "Spain", "lat": 41.3851, "lon": 2.1734},
            {"name": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "lon": 4.9041},
            {"name": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050},
            {"name": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357},
            {"name": "Bangkok", "country": "Thailand", "lat": 13.7563, "lon": 100.5018}
        ]
        
        categories = ["Museum", "Restaurant", "Park", "Monument", "Gallery", "Theater", 
                     "Market", "Beach", "Temple", "Castle", "Bridge", "Square"]
        
        attractions_per_city = count // len(cities)
        
        for city in cities:
            city_attractions = self._get_city_attractions(
                city["name"], city["country"], city["lat"], city["lon"], 
                attractions_per_city
            )
            attractions.extend(city_attractions)
        
        while len(attractions) < count:
            city = np.random.choice(cities)
            extra_attraction = self._generate_single_attraction(
                city["name"], city["country"], city["lat"], city["lon"]
            )
            attractions.append(extra_attraction)
        
        return attractions[:count]
    
    def _get_city_attractions(self, city: str, country: str, lat: float, lon: float, count: int) -> List[Attraction]:
        """Get attractions for a specific city using Overpass API (OpenStreetMap)"""
        attractions = []
        
        try:
            overpass_url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json][timeout:25];
            (
              node["tourism"~"attraction|museum|gallery|monument|castle|viewpoint|artwork"]
                  (around:5000,{lat},{lon});
              way["tourism"~"attraction|museum|gallery|monument|castle|viewpoint|artwork"]
                  (around:5000,{lat},{lon});
            );
            out center meta;
            """
            
            response = requests.get(overpass_url, params={'data': query}, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                elements = data.get('elements', [])
                
                for element in elements[:count]:
                    if 'tags' in element:
                        tags = element['tags']
                        name = tags.get('name', f"Attraction in {city}")
                        
                        if 'lat' in element and 'lon' in element:
                            attraction_lat = element['lat']
                            attraction_lon = element['lon']
                        elif 'center' in element:
                            attraction_lat = element['center']['lat']
                            attraction_lon = element['center']['lon']
                        else:
                            attraction_lat = lat + np.random.uniform(-0.1, 0.1)
                            attraction_lon = lon + np.random.uniform(-0.1, 0.1)
                        
                        attraction = self._create_attraction_from_osm(
                            city, country, name, tags, attraction_lat, attraction_lon, len(attractions)
                        )
                        attractions.append(attraction)
                        
                        if len(attractions) >= count:
                            break
            
            while len(attractions) < count:
                attraction = self._generate_single_attraction(city, country, lat, lon)
                attractions.append(attraction)
                
        except Exception as e:
            logger.warning(f"API call failed for {city}, generating mock data: {e}")
            for i in range(count):
                attraction = self._generate_single_attraction(city, country, lat, lon)
                attractions.append(attraction)
        
        return attractions
    
    def _create_attraction_from_osm(self, city: str, country: str, name: str, tags: dict, 
                                   lat: float, lon: float, idx: int) -> Attraction:
        """Create attraction from OpenStreetMap data"""
        
        tourism_type = tags.get('tourism', 'attraction')
        category_map = {
            'museum': 'Museum',
            'gallery': 'Gallery',
            'monument': 'Monument',
            'castle': 'Castle',
            'attraction': 'Attraction',
            'viewpoint': 'Viewpoint',
            'artwork': 'Gallery'
        }
        category = category_map.get(tourism_type, 'Attraction')
        
        description = self._generate_description(name, category, tags, city)
        
        cost_ranges = {
            'Museum': (8, 25),
            'Gallery': (5, 20),
            'Monument': (0, 15),
            'Castle': (10, 30),
            'Attraction': (5, 25),
            'Viewpoint': (0, 5)
        }
        min_cost, max_cost = cost_ranges.get(category, (0, 20))
        avg_cost = np.random.uniform(min_cost, max_cost)
        
        return Attraction(
            id=f"{city.replace(' ', '')}_{idx:03d}",
            city=city,
            country=country,
            name=name,
            description=description,
            category=category,
            avg_cost_usd=round(avg_cost, 2),
            rating=round(np.random.uniform(3.5, 4.8), 1),
            latitude=lat,
            longitude=lon,
            address=tags.get('addr:full', f"{city}, {country}"),
            opening_hours=tags.get('opening_hours', "9:00-17:00"),
            website=tags.get('website', ''),
            phone=tags.get('phone', ''),
            tags=self._extract_tags(tags),
            image_url=f"https://picsum.photos/400/300?random={idx}",
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def _generate_single_attraction(self, city: str, country: str, base_lat: float, base_lon: float) -> Attraction:
        """Generate a single realistic attraction"""
        
        categories = ["Museum", "Restaurant", "Park", "Monument", "Gallery", "Theater", 
                     "Market", "Beach", "Temple", "Castle", "Bridge", "Square"]
        category = np.random.choice(categories)
        
        name_templates = {
            "Museum": [f"{city} National Museum", f"Museum of {city} History", f"{city} Art Museum"],
            "Restaurant": [f"Cafe {city}", f"{city} Bistro", f"Traditional {city} Kitchen"],
            "Park": [f"Central Park {city}", f"{city} Gardens", f"Royal Park of {city}"],
            "Monument": [f"{city} Memorial", f"Victory Monument", f"Historic {city} Tower"],
            "Gallery": [f"{city} Art Gallery", f"Contemporary {city}", f"Modern Art {city}"],
            "Theater": [f"{city} Opera House", f"Royal Theater {city}", f"{city} Playhouse"],
            "Market": [f"{city} Grand Market", f"Traditional {city} Bazaar", f"Local {city} Market"],
            "Beach": [f"{city} Beach", f"Golden Beach {city}", f"{city} Waterfront"],
            "Temple": [f"Sacred Temple of {city}", f"{city} Shrine", f"Ancient {city} Temple"],
            "Castle": [f"{city} Castle", f"Royal Palace {city}", f"Historic {city} Fortress"],
            "Bridge": [f"{city} Bridge", f"Historic {city} Crossing", f"Grand Bridge {city}"],
            "Square": [f"{city} Square", f"Central Plaza {city}", f"Historic {city} Plaza"]
        }
        
        name = np.random.choice(name_templates.get(category, [f"{category} in {city}"]))
        
        description = self._generate_description(name, category, {}, city)
        
        cost_ranges = {
            "Museum": (8, 25), "Restaurant": (15, 50), "Park": (0, 10),
            "Monument": (0, 15), "Gallery": (5, 20), "Theater": (20, 80),
            "Market": (0, 5), "Beach": (0, 10), "Temple": (0, 8),
            "Castle": (10, 30), "Bridge": (0, 5), "Square": (0, 0)
        }
        min_cost, max_cost = cost_ranges.get(category, (0, 20))
        avg_cost = np.random.uniform(min_cost, max_cost)
        
        lat = base_lat + np.random.uniform(-0.1, 0.1)
        lon = base_lon + np.random.uniform(-0.1, 0.1)
        
        idx = len(self.attractions)
        
        return Attraction(
            id=f"{city.replace(' ', '')}_{idx:03d}",
            city=city,
            country=country,
            name=name,
            description=description,
            category=category,
            avg_cost_usd=round(avg_cost, 2),
            rating=round(np.random.uniform(3.5, 4.8), 1),
            latitude=lat,
            longitude=lon,
            address=f"{name}, {city}, {country}",
            opening_hours=self._generate_opening_hours(category),
            website=f"https://www.{name.lower().replace(' ', '')}.com",
            phone=self._generate_phone(),
            tags=self._generate_tags(category, city),
            image_url=f"https://picsum.photos/400/300?random={idx}",
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def _generate_description(self, name: str, category: str, tags: dict, city: str) -> str:
        """Generate realistic descriptions for attractions"""
        
        descriptions = {
            "Museum": [
                f"A world-class museum in {city} featuring extensive collections of art, history, and culture.",
                f"Home to {city}'s most precious artifacts and historical treasures spanning centuries.",
                f"An iconic cultural institution showcasing the rich heritage and artistic legacy of {city}."
            ],
            "Gallery": [
                f"A contemporary art gallery featuring works by local and international artists in {city}.",
                f"Showcasing modern and contemporary art in the heart of {city}.",
                f"A premier destination for art lovers visiting {city}."
            ],
            "Monument": [
                f"A historic monument commemorating important events in {city}'s history.",
                f"An iconic landmark representing the cultural heritage of {city}.",
                f"A significant historical site that tells the story of {city}."
            ],
            "Restaurant": [
                f"Authentic local cuisine served in a traditional atmosphere in {city}.",
                f"A culinary journey through the flavors and traditions of {city}.",
                f"Experience the finest dining and local specialties of {city}."
            ],
            "Park": [
                f"A beautiful green space in the heart of {city}, perfect for relaxation and recreation.",
                f"Urban oasis offering peaceful walks and stunning views in {city}.",
                f"A popular recreational area where locals and tourists enjoy nature in {city}."
            ],
            "Castle": [
                f"A magnificent historical castle showcasing medieval architecture in {city}.",
                f"Former royal residence now open to visitors in {city}.",
                f"Impressive fortress with centuries of history in {city}."
            ]
        }
        
        return np.random.choice(descriptions.get(category, [
            f"A must-visit attraction in {city} offering unique experiences and cultural insights."
        ]))
    
    def _generate_opening_hours(self, category: str) -> str:
        """Generate realistic opening hours based on category"""
        hours_map = {
            "Museum": "9:00-17:00",
            "Gallery": "10:00-18:00", 
            "Restaurant": "11:00-22:00",
            "Park": "6:00-20:00",
            "Theater": "19:00-23:00",
            "Market": "8:00-16:00"
        }
        return hours_map.get(category, "9:00-17:00")
    
    def _generate_phone(self) -> str:
        """Generate a realistic phone number"""
        return f"+{np.random.randint(1, 99)}-{np.random.randint(100, 999)}-{np.random.randint(1000000, 9999999)}"
    
    def _generate_tags(self, category: str, city: str) -> List[str]:
        """Generate relevant tags for the attraction"""
        base_tags = [category.lower(), city.lower(), "tourist attraction"]
        
        category_tags = {
            "Museum": ["history", "culture", "art", "educational"],
            "Gallery": ["art", "contemporary", "exhibitions", "culture"],
            "Restaurant": ["food", "dining", "local cuisine", "culinary"],
            "Park": ["nature", "outdoor", "recreation", "green space"],
            "Monument": ["historical", "heritage", "landmark", "memorial"],
            "Castle": ["medieval", "architecture", "royal", "fortress"]
        }
        
        specific_tags = category_tags.get(category, ["attraction", "sightseeing"])
        return base_tags + specific_tags
    
    def _extract_tags(self, osm_tags: dict) -> List[str]:
        """Extract relevant tags from OSM data"""
        relevant_keys = ['amenity', 'tourism', 'historic', 'building', 'leisure']
        tags = []
        
        for key in relevant_keys:
            if key in osm_tags:
                tags.append(osm_tags[key])
        
        return tags
    
    def add_attractions(self, attractions: List[Attraction]) -> None:
        """
        Add attractions to the vector database
        
        Args:
            attractions: List of Attraction objects to add
        """
        logger.info(f"Adding {len(attractions)} attractions to the database...")
        
        descriptions = [attr.description for attr in attractions]
        embeddings = self.model.encode(descriptions)
        
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        
        start_idx = len(self.attractions)
        for i, attraction in enumerate(attractions):
            self.attractions.append(attraction)
            self.id_to_idx[attraction.id] = start_idx + i
        
        logger.info(f"Successfully added {len(attractions)} attractions. Total: {len(self.attractions)}")
    
    def search_similar(self, query: str, k: int = 10, category_filter: Optional[str] = None, 
                      city_filter: Optional[str] = None, max_cost: Optional[float] = None) -> List[Dict]:
        """
        Search for similar attractions based on query
        
        Args:
            query: Search query text
            k: Number of results to return
            category_filter: Filter by category
            city_filter: Filter by city
            max_cost: Maximum cost filter
            
        Returns:
            List of matching attractions with similarity scores
        """
        if len(self.attractions) == 0:
            return []
        
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), len(self.attractions))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  
                continue
                
            attraction = self.attractions[idx]
            
            if category_filter and attraction.category.lower() != category_filter.lower():
                continue
            if city_filter and attraction.city.lower() != city_filter.lower():
                continue
            if max_cost and attraction.avg_cost_usd > max_cost:
                continue
            
            result = asdict(attraction)
            result['similarity_score'] = float(score)
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_by_id(self, attraction_id: str) -> Optional[Dict]:
        """Get attraction by ID"""
        if attraction_id in self.id_to_idx:
            idx = self.id_to_idx[attraction_id]
            return asdict(self.attractions[idx])
        return None
    
    def get_by_city(self, city: str) -> List[Dict]:
        """Get all attractions in a city"""
        results = []
        for attraction in self.attractions:
            if attraction.city.lower() == city.lower():
                results.append(asdict(attraction))
        return results
    
    def get_by_category(self, category: str) -> List[Dict]:
        """Get all attractions in a category"""
        results = []
        for attraction in self.attractions:
            if attraction.category.lower() == category.lower():
                results.append(asdict(attraction))
        return results
    
    def save_database(self, filepath: str) -> None:
        """Save the entire database to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'attractions': [asdict(attr) for attr in self.attractions],
            'id_to_idx': self.id_to_idx,
            'dimension': self.dimension
        }
        
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        with open(f"{filepath}.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Database saved to {filepath}")
    
    def load_database(self, filepath: str) -> None:
        """Load database from disk"""
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        with open(f"{filepath}.json", 'r') as f:
            data = json.load(f)
        
        self.attractions = [Attraction(**attr_data) for attr_data in data['attractions']]
        self.id_to_idx = data['id_to_idx']
        self.dimension = data['dimension']
        
        logger.info(f"Database loaded from {filepath}. {len(self.attractions)} attractions available.")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        if not self.attractions:
            return {"total_attractions": 0}
        
        cities = set(attr.city for attr in self.attractions)
        categories = set(attr.category for attr in self.attractions)
        avg_rating = np.mean([attr.rating for attr in self.attractions])
        avg_cost = np.mean([attr.avg_cost_usd for attr in self.attractions])
        
        return {
            "total_attractions": len(self.attractions),
            "cities": len(cities),
            "categories": len(categories),
            "avg_rating": round(avg_rating, 2),
            "avg_cost_usd": round(avg_cost, 2),
            "city_list": sorted(list(cities)),
            "category_list": sorted(list(categories))
        }

class PlaceFinder:
    """
    High-level interface for the travel recommendation system
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize PlaceFinder with optional database path"""
        self.db = TravelVectorDB()
        
        if db_path and os.path.exists(f"{db_path}.json"):
            self.db.load_database(db_path)
            logger.info("Loaded existing database")
        else:
            logger.info("Generating new attraction database...")
            attractions = self.db.generate_realistic_data(200)
            self.db.add_attractions(attractions)
            
            if db_path:
                self.db.save_database(db_path)
                logger.info(f"Database saved to {db_path}")
    
    def find_places(self, query: str, limit: int = 10, **filters) -> List[Dict]:
        """
        Find places based on natural language query
        
        Example queries:
        - "museums about ancient history"
        - "romantic restaurants in Paris"
        - "outdoor activities under $20"
        """
        return self.db.search_similar(query, k=limit, **filters)
    
    def get_place_details(self, place_id: str) -> Optional[Dict]:
        """Get detailed information about a specific place"""
        return self.db.get_by_id(place_id)
    
    def explore_city(self, city: str) -> List[Dict]:
        """Get all attractions in a specific city"""
        return self.db.get_by_city(city)
    
    def browse_category(self, category: str) -> List[Dict]:
        """Browse attractions by category"""
        return self.db.get_by_category(category)
    
    def get_recommendations(self, preferences: Dict) -> List[Dict]:
        """
        Get personalized recommendations based on user preferences
        
        Args:
            preferences: Dict with keys like 'interests', 'budget', 'location'
        """
        query_parts = []
        
        if 'interests' in preferences:
            query_parts.append(preferences['interests'])
        if 'budget' in preferences and preferences['budget'] == 'low':
            query_parts.append("affordable budget-friendly")
        if 'type' in preferences:
            query_parts.append(preferences['type'])
        
        query = " ".join(query_parts) if query_parts else "popular attractions"
        
        filters = {}
        if 'city' in preferences:
            filters['city_filter'] = preferences['city']
        if 'max_budget' in preferences:
            filters['max_cost'] = preferences['max_budget']
        
        return self.find_places(query, **filters)

# Example usage
if __name__ == "__main__":
    place_finder = PlaceFinder("travel_attractions_db")
    
    print("=== Museum Search ===")
    museums = place_finder.find_places("museums with ancient artifacts", limit=5)
    for museum in museums:
        print(f"- {museum['name']} in {museum['city']} (Score: {museum['similarity_score']:.3f})")
    
    print("\n=== Restaurant Search ===")
    restaurants = place_finder.find_places("romantic dinner restaurants", limit=5, category_filter="Restaurant")
    for restaurant in restaurants:
        print(f"- {restaurant['name']} in {restaurant['city']} (${restaurant['avg_cost_usd']})")
    
    print("\n=== Budget Search ===")
    budget_places = place_finder.find_places("free attractions and monuments", max_cost=10)
    for place in budget_places[:5]:
        print(f"- {place['name']} in {place['city']} (${place['avg_cost_usd']})")
    
    print("\n=== Database Statistics ===")
    stats = place_finder.db.get_statistics()
    print(json.dumps(stats, indent=2))