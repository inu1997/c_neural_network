LIB_CSRCS:= neural_network.c neural_network_elite.c neural_network_util.c
LIB_COBJS:= $(LIB_CSRCS:.c=.o)

CC:=gcc
CFLAGS:= -I. -lm -fPIC
LDFLAGS:= -L.

TARGETS:=example1 example2 libnn.so

.PHONY: all
all: $(TARGETS)

.PHONY: libnn.so
libnn.so: $(LIB_COBJS)
	@$(CC) $(CFLAGS) $(LDFLAGS) -shared $^ -o $@

.PHONY: example1
example1: example/example1.o libnn.so
	@echo "Linking $@ ..."
	@$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-rpath=. -lnn $^ -o $@

.PHONY: example2
example2: example/example2.o libnn.so
	@echo "Linking $@ ..."
	@$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-rpath=. -lnn $^ -o $@

%.o: %.c
	@echo "Compiling $@ ..."
	@$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(LIB_COBJS) example/example1.o example/example2.o
	rm -f $(TARGETS)

