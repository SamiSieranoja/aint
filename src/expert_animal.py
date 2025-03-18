import clips
# pip install clipspy


def create_expert_system():
    env = clips.Environment()
    
    # Define templates for facts
    env.build("""
    (deftemplate animal
        (slot body-covering)
        (slot legs)
        (slot can-fly (type SYMBOL))
    )
    """)
    
    # Define rules to classify animal types
    env.build("""
    (defrule classify-mammal
        (animal (body-covering fur))
        =>
        (assert (animal-type mammal))
    )
    """)
    
    env.build("""
    (defrule classify-lizard
        (animal (body-covering scales))
        =>
        (assert (animal-type lizard))
    )
    """)
    
    env.build("""
    (defrule classify-bird
        (animal (body-covering feathers))
        =>
        (assert (animal-type bird))
    )
    """)
    
    # Define rules for species classification based on type
    env.build("""
    (defrule classify-lion
        (animal-type mammal)
        (animal (legs 4))
        =>
        (assert (species lion))
    )
    """)
    
    env.build("""
    (defrule classify-jeti
        (animal-type mammal)
        (animal (legs 2)  (can-fly no))
        =>
        (assert (species jeti))
    )
    """)
    
    env.build("""
    (defrule classify-eagle
        (animal-type bird)
        (animal (can-fly yes))
        =>
        (assert (species eagle))
    )
    """)
    
    env.build("""
    (defrule classify-penguin
        (animal-type bird)
        (animal (can-fly no))
        =>
        (assert (species penguin))
    )
    """)
    
    env.build("""
    (defrule classify-iguana
        (animal-type lizard)
        (animal (legs 4))
        =>
        (assert (species iguana))
    )
    """)
    
    return env

def run_expert_system(env):
    print("Enter the following details to classify the animal:")
    body_covering = input("Body covering (fur, scales, feathers): ")
    legs = input("Number of legs (2, 4): ")
    can_fly = input("Can it fly? (yes/no): ")
    
    # Assert facts into the system
    env.assert_string(f"(animal (body-covering {body_covering}) (legs {legs}) (can-fly {can_fly})  )")
    env.run()
        
    # Retrieve inferred facts
    for fact in env.facts():
        if 'animal-type' in str(fact):
            print("Animal Type:", fact)
        if 'species' in str(fact):
            print("Species Identified:", fact)

if __name__ == "__main__":
    env = create_expert_system()
    run_expert_system(env)

