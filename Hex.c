#include <stdio.h>
#include <stdlib.h>

#ifndef BOARD_DIM
    #define BOARD_DIM 8
#endif

int neighbors[] = {-(BOARD_DIM+2) + 1, -(BOARD_DIM+2), -1, 1, (BOARD_DIM+2), (BOARD_DIM+2) - 1};

struct hex_game {
	int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
	int open_positions[BOARD_DIM*BOARD_DIM];
	int number_of_open_positions;
	int moves[BOARD_DIM*BOARD_DIM];
	int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void hg_init(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM+2; ++i) {
		for (int j = 0; j < BOARD_DIM+2; ++j) {
			hg->board[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			hg->board[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;

			if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
				hg->open_positions[(i-1)*BOARD_DIM + j - 1] = i*(BOARD_DIM + 2) + j;
			}

			if (i == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			}
			
			if (j == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;
			}
		}
	}
	hg->number_of_open_positions = BOARD_DIM*BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position) 
{
	hg->connected[position*2 + player] = 1;

	if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
			if (hg_connect(hg, player, neighbor)) {
				return 1;
			}
		}
	}
	return 0;
}

int hg_winner(struct hex_game *hg, int player, int position)
{
	for (int i = 0; i < 3; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->connected[neighbor*2 + player]) {
			return hg_connect(hg, player, position);
		}
	}
	return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player)
{
	int random_empty_position_index = rand() % hg->number_of_open_positions;

	int empty_position = hg->open_positions[random_empty_position_index];

	hg->board[empty_position * 2 + player] = 1;

	hg->moves[BOARD_DIM*BOARD_DIM - hg->number_of_open_positions] = empty_position;

	hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions-1];

	hg->number_of_open_positions--;

	return empty_position;
}

void hg_place_piece_based_on_tm_input(struct hex_game *hg, int player)
{
	printf("TM!\n");
}

int hg_full_board(struct hex_game *hg)
{
	return hg->number_of_open_positions == 0;
}

void hg_print(struct hex_game *hg, char *board_str)
{   
	int index = 0;
	for (int i = 0; i < BOARD_DIM; ++i) {
		for (int j = 0; j < BOARD_DIM; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				board_str[index++] = '1';
				printf("1");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				board_str[index++] = '2';
				printf("2");
			} else {
				board_str[index++] = '0';
				printf("0");
			}
		}
	}
	board_str[index] = '\0';
}

int main() {
	struct hex_game hg;

    FILE *fpt0;
    FILE *fpt1;	
    fpt0 = fopen("player0_8_5.csv", "a");
    fpt1 = fopen("player1_8_5.csv", "a");
	int winner = -1;
    int a = 0;
	for (int game = 0; game < 30000000; ++game) {
		hg_init(&hg); // Initialize game state

        char board_str[BOARD_DIM * (BOARD_DIM * 2 + 1) + 1];

		int player = 0; // Start with player 0
		int total_moves = 0;

        while (!hg_full_board(&hg)) {
            int position = hg_place_piece_randomly(&hg, player);
            hg.moves[total_moves++] = position; // Store the move

            if (hg_winner(&hg, player, position)) {
                winner = player; // Declare winner
                break; // Exit loop if there's a winner
            }

            player = 1 - player; // Switch players
        }

        // If a winner was found and we have enough moves, backtrack to two moves before the end
        if (hg.number_of_open_positions >= 35 && winner != -1) {
            // Reset the board
            hg_init(&hg);
            for (int i = 0; i < total_moves - 2; ++i) {
                int pos = hg.moves[i];
                hg.board[pos * 2 + (i % 2)] = 1; // Re-apply moves up to two before end
            }

            // Get the board state string before the last two moves
            char board_str[BOARD_DIM * (BOARD_DIM * 2 + 1) + 1];
            hg_print(&hg, board_str);

            // Determine final winner by replaying the last two moves
            if (winner == -1) { // If no winner yet
                for (int i = total_moves - 2; i < total_moves; ++i) {
                    int pos = hg.moves[i];
                    hg.board[pos * 2 + (i % 2)] = 1; // Apply last two moves
                    if (hg_winner(&hg, i % 2, pos)) {
                        winner = i % 2; // Confirm the winner
                        break;
                    }
                }
            }

            // Save the board state and winner
            if (winner == 0) {
                fprintf(fpt0, "%s\n", board_str); // Record board state for Player 0
                printf("Player 0 winner: %s\n", board_str); // Optional debug print
            } else if (winner == 1 && a < 100000) {
                fprintf(fpt1, "%s\n", board_str); // Record board state for Player 1
                printf("Player 1 winner: %s\n", board_str); // Optional debug print
                ++a;
            }
            if (a > 100000) {
                abort(); 
            }
        }
        

        // Reset winner for the next game
        winner = -1;
    }
    
    fclose(fpt0); // Close the file when done
    fclose(fpt1); // Close the file when done
    return 0;
}