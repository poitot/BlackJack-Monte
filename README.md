# BlackJack-Monte
Project created for honours dissertation using reinforcement learning to play black jack including side betting using modified version of the Open AI Gym Blackjack enviroment 

Created to explore using reinforcement learning to determine a blackajck strategy using side bets

Actions supported: Hit, Stand, Double
Side bets: 21 + 3: player wins by creating a poker hand using their starting hand and the dealers showing card, such as three of a kind (A, A, A), all cards follow one another in sequence (straight (3, 4, 5)) or if all cards suits match (flush)

To run clone the repository and run the monte_bj_2,py file, the number of runs and epsilon values can be configured in the main function.

This project makes use of on-policy monte carlo methods using a package created  by Ray Zhang which can be found @ https://github.com/OneRaynyDay/RLEngine

TODO: ui*
