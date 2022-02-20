from manim import *

from neural_network import NeuralNetworkMobject



class CreateSquare(Scene):
    """Tutorial Class"""

    def construct(self):
        sq = Square(
            side_length=5, stroke_color=GREEN, fill_color=BLUE, fill_opacity=0.75
        )
        self.play(Create(sq), run_time=3)
        self.wait()


class Testing(Scene):
    """Testing class"""

    def construct(self):
        name = Tex(r"\^{y}$ = e^{t}\cdot$").to_edge(UL, buff=0.5)
        sq = Square(side_length=0.5, fill_color=GREEN).shift(LEFT * 3)
        tri = Triangle().scale(0.6).to_edge(DR)
        self.play(Write(name))
        self.play(DrawBorderThenFill(sq), run_time=2)
        self.play(Create(tri))
        self.wait()


class Circles(Scene):
    """Try array of circles"""

    def construct(self):
        num_circles = 10
        circles = [
            Circle(fill_color=PINK, fill_opacity=0.5, radius=0.1)
            for i in range(num_circles)
        ]  # create a circle
        for circle_index, circle in enumerate(circles):
            circle.move_to([circle_index / 4, 0, 0])

        self.play(*(Create(circle) for circle in circles), run_time=2)


class NeuralNetworkScene(Scene):
    """Trying out neural network object"""

    def construct(self):
        visualNetwork = (
            NeuralNetworkMobject([7, 6, 5, 5], BLUE).rotate(PI / 2).shift([-0.5, 0, 0])
        )
        languageNetwork = NeuralNetworkMobject([6, 5, 4, 4], GREEN).rotate(PI / 2)
        visualNetwork.neuron_fill_color = BLUE
        languageNetwork.next_to(visualNetwork, RIGHT)

        image = ImageMobject("dog.jpeg").scale(0.3)
        image.next_to(visualNetwork, DOWN)

        dogText = Tex('"A picture of a dog."').scale(0.5)
        dogText.next_to(languageNetwork, DOWN)
        # myNetwork.label_inputs('x')
        # myNetwork.label_outputs('\hat{y}')
        # myNetwork.label_outputs_text("isPedestrian")

        import copy

        last_layer1 = copy.deepcopy(visualNetwork.layers[-1])
        last_layer2 = copy.deepcopy(languageNetwork.layers[-1])

        last_layer1.rotate(-PI / 2)

        last_layer2.rotate(-PI / 2)

        last_layer2.next_to(last_layer1, RIGHT, buff=1)

        self.play(Write(dogText), FadeIn(image))
        self.play(Write(visualNetwork), Write(languageNetwork))
        self.play(
            FadeOut(dogText),
            FadeOut(image),
            FadeOut(*visualNetwork.layers[:-1]),
            FadeOut(*languageNetwork.layers[:-1]),
            FadeOut(*visualNetwork.edge_groups),
            FadeOut(*languageNetwork.edge_groups),
        )
        self.play(
            ReplacementTransform(visualNetwork.layers[-1], last_layer1),
            ReplacementTransform(languageNetwork.layers[-1], last_layer2),
        )
        description = Tex("Now to make the dimensions match!")
        self.play(Write(description))
        self.play(FadeOut(description))

        def animate_shared_embedding(source, color, size, text, shared_embedding_size):

            wI = MobjectMatrix(
                [
                    [
                        Circle(
                            fill_color=WHITE,
                            stroke_color=WHITE,
                            fill_opacity=1,
                            radius=0.1,
                        )
                        for _column in range(size)
                    ]
                    for _row in range(shared_embedding_size)
                ]
            )
            wI.to_edge(DL)
            multiplication = Tex("X").scale(0.5)
            multiplication.next_to(wI, RIGHT, buff=0.75)
            eI = MobjectMatrix(
                [
                    [
                        Circle(
                            fill_color=color,
                            stroke_color=color,
                            fill_opacity=1,
                            radius=0.07,
                        )
                    ]
                    for _ in range(size)
                ]
            )
            eI.next_to(multiplication, RIGHT, buff=0.75)
            wIDescription = (
                Tex("Learned embedding matrix for %s embeddings" % text)
                .scale(0.5)
                .next_to(wI, UP)
            )
            equals = Tex("=").scale(0.5).next_to(eI, RIGHT)

            eShared = MobjectMatrix(
                [
                    [
                        Circle(
                            fill_color=color,
                            stroke_color=color,
                            fill_opacity=1,
                            radius=0.07,
                        )
                    ]
                    for _ in range(3)
                ]
            )

            eShared.next_to(equals, RIGHT)

            self.play(
                ReplacementTransform(source, eI),
                Write(multiplication),
                Write(wI),
                Write(wIDescription),
                Write(equals),
                Write(eShared),
            )
            self.wait()
            self.play(
                FadeOut(eI),
                FadeOut(multiplication),
                FadeOut(wI),
                FadeOut(wIDescription),
                FadeOut(equals),
            )
            return eShared

        vShared = animate_shared_embedding(last_layer1, BLUE, 5, "visual", 3)
        vSharedTop = MobjectMatrix(
            [
                [Circle(fill_color=BLUE, stroke_color=BLUE, fill_opacity=1, radius=0.1)]
                for _ in range(3)
            ]
        ).to_edge(UL)
        self.play(ReplacementTransform(vShared, vSharedTop))
        lShared = animate_shared_embedding(last_layer2, GREEN, 4, "language", 3)
        dot = Tex(r"$\cdot$").next_to(vSharedTop, RIGHT)
        lSharedTop = MobjectMatrix(
            [
                [
                    Circle(
                        fill_color=GREEN, stroke_color=GREEN, fill_opacity=1, radius=0.1
                    )
                ]
                for _ in range(3)
            ]
        ).next_to(dot, RIGHT)
        self.play(ReplacementTransform(lShared, lSharedTop), Write(dot))

        def norm(embedding):
            tex1 = Tex("Norm(")
            embedding_copy = copy.deepcopy(embedding)
            tex2 = Tex(")")
            frac = Tex(r"$\frac{\phantom{1}}{\phantom{1}}$")
            frac.next_to(embedding, DOWN)
            embedding_copy.next_to(frac, DOWN)
            tex1.next_to(embedding_copy, LEFT)
            tex2.next_to(embedding_copy, RIGHT)
            return VGroup(tex1, embedding_copy, tex2, frac)

        vSharedCentered = copy.deepcopy(vSharedTop)
        vSharedCentered.shift(RIGHT * 1.5)
        dotCentered = copy.deepcopy(dot)
        dotCentered.next_to(vSharedCentered, RIGHT, buff=1.5)
        lSharedCentered = copy.deepcopy(lSharedTop)
        lSharedCentered.next_to(dotCentered, RIGHT, buff=1.5)

        self.play(
            ReplacementTransform(vSharedTop, vSharedCentered),
            ReplacementTransform(lSharedTop, lSharedCentered),
            ReplacementTransform(dot, dotCentered),
        )

        
        cossineSimilarity = Tex("= Cosine Similarity Score").next_to(lSharedCentered, RIGHT)
        normElements = norm(vSharedCentered) + norm(lSharedCentered) + VGroup(cossineSimilarity)
        self.play(Write(normElements))

        def vector_pair():
            vg = VGroup()
            v1 = MobjectMatrix(
                [
                    [
                        Circle(
                            fill_color=BLUE,
                            stroke_color=BLUE,
                            fill_opacity=1,
                            radius=0.1,
                        )
                    ]
                    for _ in range(3)
                ]
            )
            v2 = MobjectMatrix(
                [
                    [
                        Circle(
                            fill_color=GREEN,
                            stroke_color=GREEN,
                            fill_opacity=1,
                            radius=0.1,
                        )
                    ]
                    for _ in range(3)
                ]
            )
            dot = Tex(r"$\cdot$")
            dot.next_to(v1, RIGHT)
            v2.next_to(dot, RIGHT)
            vg.add(v1)
            vg.add(dot)
            vg.add(v2)
            return vg.scale(0.1)

        pairwiseMatrix = MobjectMatrix(
            [[vector_pair() for _ in range(4)] for _ in range(4)]
        )

        self.play(
            FadeOut(vSharedCentered),
            FadeOut(lSharedCentered),
            FadeOut(dotCentered),
            FadeOut(normElements),
            ReplacementTransform(vSharedTop, pairwiseMatrix[0][0][0]),
            ReplacementTransform(dot, pairwiseMatrix[0][0][1]),
            ReplacementTransform(lSharedTop, pairwiseMatrix[0][0][2]),
            Write(pairwiseMatrix),
        )


        Tex(r"\^{y}$ = e^{t}\cdot$")
        labelMatrix = Matrix([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]).scale(.7)
        labelMatrix.next_to(pairwiseMatrix, DOWN)
        yHat = Tex(r"\^{y}$ = $").next_to(labelMatrix, LEFT).scale(1.2)
        movedPairwiseMatrix = pairwiseMatrix.to_edge(UP)
        xHat = Tex(r"\^{x} $ = e^{t} \cdot $").next_to(movedPairwiseMatrix, LEFT).scale(1.2)

        self.play(Write(xHat), Write(yHat), ReplacementTransform(pairwiseMatrix, movedPairwiseMatrix), Write(labelMatrix))