{'hotel': {'hotel-pricerange', 'hotel-area', 'hotel-ref', 'hotel-postcode', 'hotel-address', 'hotel-stars', 'hotel-internet', 'hotel-parking', 'hotel-name', 'hotel-phone', 'hotel-type'},
 # Hotel-Ref is the reference number, returned only when the user provides the number of people. It is tied with availability
 # Need to ask if not provided: hotel-pricerange, hotel-area
 'restaurant': {'restaurant-postcode', 'restaurant-food', 'restaurant-area', 'restaurant-pricerange', 'restaurant-ref', 'restaurant-address', 'restaurant-name', 'restaurant-phone'}}
def given(slots):
    return [slot for slot, value in slots if value != '?']

def predict_to_be_retrieved(user_act_types, user_slots_per_act_types):
    to_be_retrieved = []
    # Always add availability for domains
    if 'Hotel-Inform' in user_act_types or 'Hotel-Request' in user_act_types:
        to_be_retrieved.append(['hotel-availability'])
    if 'Restaurant-Inform' in user_act_types or 'Restaurant-Request' in user_act_types:
        to_be_retrieved.append(['restaurant-availability'])
    
    # When the booking is finalized for a hotel
    if 'Hotel-Inform' in user_act_types:
        hot_inf = given(user_slots_per_act_types['Hotel-Inform'])
    if 'bookpeople' in hot_inf:
        hotel_booked = True
        to_be_retrieved.append('hotel-ref')
    
    # When booking is finalized for a restaurant
    if 'Restaurant-Inform' in user_act_types:
        rest_inf = given(user_slots_per_act_types['Restaurant-Inform'])
    if 'bookpeople' in rest_inf and 'bookday' in rest_inf and 'booktime' in rest_inf:
        restauarant_booked = True
        to_be_retrieved.append('restaurant-ref')
    
    # When we need to ask for further information about a hotel
    if not hotel_booked and 'Hotel-Inform' in user_act_types:
        if 'hotel-pricerange' not in hot_inf:
            to_be_retrieved.append('hotel-pricerange')
        if 'hotel-type' not in hot_inf:
            to_be_retrieved.append('hotel-type')
    
    # When we need to ask for further information about a restaurant
    if not restauarant_booked and 'Restaurant-Inform' in user_act_types:
        if not restauarant_booked and 'food' in rest_inf and 'pricerange' in rest_inf:
            to_be_retrieved.append('restaurant-name')
    
    
    #....... TOO TEDIOUS
    
    restaurant_infos = ['name', 'area']
    known_slots = []
    for act_type in user_slots_per_act_types:
        for slot, value in user_slots_per_act_types[act_type]:
            if value == '?':
                user_slots_per_act_types[act_type].remove((slot, value))
                to_be_retrieved.append(slot)
            else:
                known_slots.append(slot)
    