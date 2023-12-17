import random

class FakeDatabase():

    def retrieve(self, key):
        return random.choice(self.to_be_retrieved[key])
    
    def retrieve_agent_response(self, agent_dialogue_act, to_be_requested, to_be_provided):
        sentence = random.choice(self.templates['greeting']) + ' '
        for slot in to_be_provided.keys():
            value = to_be_provided[slot]
            domain = slot.split('-')[0]
            name = slot.split('-')[1]
            if name in self.templates['to_be_provided']:            
                sentence += random.choice(self.templates['to_be_provided'][name]).format(**{name: value, 'domain': domain}) + ' '
            
        for slot in to_be_requested:
            if slot == 'none':
                continue
            domain = slot.split('-')[0]
            name = slot.split('-')[1]
            if name in self.templates['to_be_request']:    
                sentence += random.choice(self.templates['to_be_request'][name]).format(**{'domain': domain}) + ' '
        
        if len(to_be_provided) == 0 and len(to_be_requested) == 0 and len(agent_dialogue_act) == 0:
            sentence = "I'm sorry, I don't have any information about that."
        
        if any([act for act in agent_dialogue_act if act.startswith('general')]):
            sentence = "You're welcome!"
        return sentence
    
    def __init__(self):
        self.to_be_retrieved = {
            'hotel-internet': ['yes', 'no'],
            'hotel-parking': ['yes', 'no'],
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
        
        self.templates = {}
        self.templates['to_be_request'] = {}
        self.templates['to_be_provided'] = {}
        self.templates['greeting'] = [
            "Great!",
            "Fantastic!",
            "Awesome!",
            "Perfect!",
            "Excellent!"
            ]

        self.templates['to_be_request']['area'] = [
            "Of course! In which lovely area would you prefer {domain}?",
            "I'd be delighted to help! Could you share the charming area you have in mind for {domain}?",
            "Great choice! I'm here to assist. Could you let me know the specific area you're thinking of for {domain}?",
            "Absolutely! Before we continue, could you share the preferred area for {domain}?",
            "When it comes to the area, do you have a special location in mind for {domain}?"
        ]

        self.templates['to_be_request']['bookday'] = [
            "Let me know the day you'd like to make the booking.",
            "When would you like to book your stay?",
            "I just need to know the date for your booking.",
            "To assist you further, could you provide the specific day for your booking?",
            "But, before we proceed, could you share the date you have in mind for the booking?"
        ]

        self.templates['to_be_request']['bookpeople'] = [
            "How many people are we expecting for {domain}?",
            "Can you specify the number of people for {domain}?",
            "To better serve you, could you let me know how many people will be joining for {domain}?",
            "Before we proceed, may I know the number of people for {domain}?",
            "When it comes to the reservation for {domain}, how many people are we accommodating?"
        ]

        self.templates['to_be_request']['bookstay'] = [
            "How many days are you planning to stay?",
            "What is the intended duration of your stay?",
            "Could you specify the number of days you plan to stay?",
            "Is there a particular length of stay you have in mind?",
            "Do you have a preferred duration for your stay? If so, how many days?"
        ]

        self.templates['to_be_request']['booktime'] = [
            "For {domain}, could you please specify the preferred time for the reservation?",
            "When would you like to book for {domain}?",
            "To better accommodate your needs, could you provide the desired time for the reservation at {domain}?",
            "Before we proceed, may I know the preferred time for the reservation of {domain}?",
            "When it comes to the reservation for {domain}, what time are you thinking of?"
        ]

        self.templates['to_be_request']['food'] =  [
            "Do you have any specific preferences or dietary restrictions for the menu?",
            "Are there any particular cuisines or dishes you're in the mood for at {domain}?",
            "Could you share any specific food preferences or dietary requirements for {domain}?",
            "Is there anything specific you'd like to mention about your food preferences?",
            "Do you have any favorite cuisines or specific dishes in mind?"
        ]

        self.templates['to_be_request']['internet'] = [
            "Would you like Wi-Fi?",
            "Do you prefer having Wi-Fi?",
            "For a connected experience, do you want Wi-Fi?",
            "Is Wi-Fi a preference for you?",
            "Do you have a preference for Wi-Fi?" 
            ]

        self.templates['to_be_request']['name'] = [
            "Is there a specific place you would like to go or reserve?",
            "Do you have a particular place in mind for your reservation?",
            "Would you like to specify the name of the place for your reservation?",
            "Is there a preferred venue or location for your reservation?",
            "Do you already have a place in mind that you'd like to go to or reserve?"
        ]

        self.templates['to_be_request']['parking'] = [
            "Do you require parking for your visit?",
            "Is parking a consideration for you?",
            "Would you like information about parking options?",
            "Do you need assistance with parking arrangements?",
            "Is parking availability something you'd like to discuss or specify?"
        ]

        self.templates['to_be_request']['pricerange'] = [
            "What is your preferred price range for this reservation?",
            "Could you specify your budget or price range?",
            "Is there a particular price range you have in mind?",
            "Do you have a preferred budget for your reservation?",
            "Could you let us know your price range for the reservation?"
        ]

        self.templates['to_be_request']['stars'] = [
            "Are you looking for a place with a specific star rating?",
            "Do you have a preference for the star rating of the place?",
            "Is there a particular star rating you are aiming for?",
            "Would you like recommendations based on a certain star rating?",
            "Is the star rating of the place important to you? If so, what rating are you looking for?"
        ]

        self.templates['to_be_request']['type'] = [
            "Do you have a specific type of {domain} in mind?",
            "Are you looking for a particular style or type of {domain}?",
            "Could you specify the type of {domain} you prefer?",
            "Is there a particular theme or category of {domain} you're interested in?",
            "Do you have a preference for a specific type of {domain}?"
        ]

        self.templates['to_be_provided']['address'] = [
            "The place is located in {address}.",
            "The address is {address}.",
            "The current address is {address}.",
            "The current location is {address}.",
        ]

        self.templates['to_be_provided']['area'] = [
            "The {domain} is located in {area}.",
            "The {domain} is in {area}.",
            "The {domain} is situated in {area}.",
            "The {domain} is in the {area} area.",
        ]

        self.templates['to_be_provided']['bookday'] = [
            "You can reserve the place on {bookday}.",
            "The place is available on {bookday}.",
            "The place is open on {bookday}.",
            "The place is accepting reservations on {bookday}."
        ]

        self.templates['to_be_provided']['availability'] = [
            "The place is available.",
            "Yes, the place is available. We can proceed with the reservation.",
            "The requested place is available for your reservation.",
            "You can book the place as it is available."
        ]

        self.templates['to_be_provided']['bookpeople'] = [
            "I can make a reservation for {bookpeople} people.",
            "The place can accommodate {bookpeople} people.",
            "In the {domain} there are {bookpeople} seats available.",
        ]

        self.templates['to_be_provided']['bookstay'] = [
            "The place is available for {stay} days.",
            "The place can accommodate you for {stay} days.",
            "The place is open for {stay} days.",
        ]

        self.templates['to_be_provided']['booktime'] = [
            "The place is available at {booktime}.",
            "The place is open at {booktime}.",
            "The place is accepting reservations at {booktime}.",
            "You can book the place at {booktime}."
        ]

        self.templates['to_be_provided']['choice'] = [
            "There are {choice} options available.",
            "There are {choice} options for you to choose from.",   
            "You will have {choice} options to choose from."
        ]

        self.templates['to_be_provided']['food'] = [
            "One possible option is {food} food.",
            "You can try {food} food.",
            "They have {food} food."
        ]

        self.templates['to_be_provided']['internet'] = [
            "The place offers {internet}.",
            "{internet} is available at the place.",
            "The place has {internet}."
        ]

        self.templates['to_be_provided']['name'] = [
            "The name of the {domain} is {name}.",
            "It's called {name}.",
            "It's named {name}.",
            "The {domain} is {name}."
        ]

        self.templates['to_be_provided']['parking'] = [
            "The place has {parking} parking.",
            "{parking} parking is available.",
            "There is {parking} parking."
        ]

        self.templates['to_be_provided']['phone'] = [
            "The phone number is {phone}.",
            "The contact number is {phone}.",
            "The number is {phone}."
        ]

        self.templates['to_be_provided']['postcode'] = [
            "The postcode is {postcode}.",
            "The postal code is {postcode}.",
            "The zip code is {postcode}."
        ]

        self.templates['to_be_provided']['pricerange'] = [
            "The price range is {pricerange}.",
            "If you are interested, the budget is {pricerange}.",
            "The {domain} is {pricerange}."
        ]

        self.templates['to_be_provided']['ref'] = [
            "The reference number is {ref}.",
            "The reference is {ref}.",
            "The reference number is {ref}."
        ]

        self.templates['to_be_provided']['stars'] = [
            "The place has {stars} stars.",
            "The place is rated {stars} stars.",
            "The place is {stars} stars."
        ]

        self.templates['to_be_provided']['type'] = [
            "The place is a {type}.",
            "There is a {type} type.",
            "Is a {type} style."
        ]