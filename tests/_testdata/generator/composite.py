from typing import Union

from tests._testdata.generator.data_generator import BaseDataGenerator


class CompositeDataGenerator:
    def __init__(
        self, *generators: list[Union[BaseDataGenerator, "CompositeDataGenerator"]]
    ):
        self.generators: list[BaseDataGenerator | CompositeDataGenerator] = generators

    def __and__(self, other: Union[BaseDataGenerator, "CompositeDataGenerator"]):
        return AndGenerator(self, other)

    def __or__(self, other: Union[BaseDataGenerator, "CompositeDataGenerator"]):
        return OrGenerator(self, other)

    def flatten(self):
        flat_set = set()
        for generator in self.generators:
            if isinstance(generator, CompositeDataGenerator):
                flat_set.update(generator.flatten())
            else:
                flat_set.add(generator)
        return flat_set


class AndGenerator(CompositeDataGenerator):
    def __str__(self):
        return f"({' & '.join(str(generator) for generator in self.generators)})"


class OrGenerator(CompositeDataGenerator):
    def __str__(self):
        return f"({' | '.join(str(generator) for generator in self.generators)})"


class NotGenerator(CompositeDataGenerator):
    def __init__(self, *generators: list[BaseDataGenerator]):
        if len(generators) != 1:
            raise ValueError("NotGenerator must have exactly one generator")
        self.generator = generators[0]

    def __str__(self):
        return f"~{str(self.generator)}"


class ExactlyOneGenerator(CompositeDataGenerator):
    def __str__(self):
        return (
            f"ExactlyOne({', '.join(str(generator) for generator in self.generators)})"
        )


class AtLeastOneGenerator(CompositeDataGenerator):
    def __str__(self):
        return (
            f"AtLeastOne({', '.join(str(generator) for generator in self.generators)})"
        )
