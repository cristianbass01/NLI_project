import random

class FakeDatabase():
    def __init__(self):
        self.to_be_retrieved = {
            'hotel-internet': ['yes', 'no'],
            'hotel-parking': ['yes', 'no'],
            'hotel-type': ['hotel', 'guesthouse'],
            'hotel-phone': ['+1-123-456-7890', '+1-098-765-4321', 'hotel-none'],
            'hotel-pricerange': ['cheap', 'expensive', 'moderate'],
            'hotel-stars': ['3', '4', '5'],
            'hotel-name': ['Grand Hotel', 'Hilton', 'Holiday Inn', 'hotel-none'],
            'hotel-ref': ['B0007I1JYQ', 'B0007I1JYR', 'B0007I1JYS'],
            
            'restaurant-area': ['center', 'east', 'north', 'south', 'west'],
            'restaurant-food': ['chinese', 'english', 'french', 'indian', 'italian', 'japanese'],
            'restaurant-phone': ['+1-123-456-7890', '+1-098-765-4321', 'restaurant-none'],
            'restaurant-name': ['Django\'s', 'Jin\'s', 'Sausalitos', 'retaurant-none'],
            'restaurant-pricerange': ['cheap', 'expensive', 'moderate'],
        }

    def retrieve(self, key):
        return random.choice(self.to_be_retrieved[key])